import os
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import f1_score
from tqdm import tqdm


############################## 1. 训练 ##############################
# 1.1 数据加载
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 1.1 数据加载
def get_datasets(train_path, dev_path):
    # 1.1 读成json文件，两个列表，每个元素是一个字典
    train_data = load_json(train_path)
    dev_data = load_json(dev_path)

    # 1.2 加载成Dataset实例
    train_dataset = Dataset.from_list(train_data)
    dev_dataset = Dataset.from_list(dev_data)

    # 1.3 返回
    return train_dataset, dev_dataset

# 2. 拿到预训练的模型
def get_model_and_tokenizer(model_name='hfl/chinese-macbert-large'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return tokenizer, model


# 3. 评估指标
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    macro_f1 = f1_score(labels, preds, average='macro')
    return {'macro_f1': macro_f1}


def train():
    # 1. 读取数据集
    train_dataset, dev_dataset = get_datasets('./data/train.json', './data/dev.json')

    # 2. 拿到预训练的模型，并指定使用 GPU 2
    device = torch.device("cuda:2")  # 明确指定使用 GPU 2
    tokenizer, model = get_model_and_tokenizer()
    model.to(device)

    # 3. 把训练集和验证集转成token
    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_dev = dev_dataset.map(preprocess_function, batched=True)

    # 4. 训练参数
    logs_dir = os.path.abspath('./logs')
    results_dir = os.path.abspath('./results')
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=results_dir,
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,  # 可调为 5~10
        weight_decay=0.01,
        logging_dir=logs_dir,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        greater_is_better=True,
        report_to="tensorboard",
        save_total_limit=2,

        # 以下是新增/改进部分
        lr_scheduler_type="cosine",  # 可选 'linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant'
        warmup_ratio=0.1,  # 前 10% 训练步数用于 warmup
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # 如果2个epoch不提升，就停止
    )

    model_dir = os.path.abspath('./best_model')
    os.makedirs(model_dir, exist_ok=True)
    print("开始训练...")
    trainer.train()
    print("训练完成，保存模型中...")
    trainer.save_model(model_dir)
    print(f"模型已保存到: {model_dir}")

############################## 2. 推理与生成提交文件 ##############################
def predict():
    device = torch.device("cuda:2")  # 明确指定使用 GPU 2
    test_data = load_json('./data/test.json')
    tokenizer = AutoTokenizer.from_pretrained('./best_model')
    model = AutoModelForSequenceClassification.from_pretrained('./best_model')
    model.to(device)
    model.eval()

    results = []
    for item in tqdm(test_data):
        inputs = tokenizer(item['text'], return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_label = torch.argmax(logits, dim=1).item()
        results.append({
            'id': item['id'],
            'text': item['text'],
            'label': predicted_label
        })

    with open('submission.json', 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print("推理完成，结果已保存到 submission.json")

############################## 3. 评估准确率 ##############################
def evaluate():
    submission = {}
    with open('submission.json', 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            submission[item['id']] = item['label']

    with open('./data/test_with_label.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    correct = 0
    total = 0
    for item in test_data:
        id_ = item['id']
        true_label = item['label']
        pred_label = submission.get(id_, None)
        if pred_label is not None:
            if pred_label == true_label:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"总样本数: {total}")
    print(f"预测正确数: {correct}")
    print(f"准确率: {accuracy:.4f}")

############################## 4. pack保存结果 ##############################
def file2zip(packagePath, zipPath):
    import zipfile
    zipf = zipfile.ZipFile(zipPath, 'w', zipfile.ZIP_DEFLATED)
    for path, dirNames, fileNames in os.walk(packagePath):
        fpath = path.replace(packagePath, '')
        for name in fileNames:
            fullName = os.path.join(path, name)
            name = fpath + '/' + name
            zipf.write(fullName, name)
    zipf.close()

def pack():
    packagePath = './'
    zipPath = './output.zip'
    if os.path.exists(zipPath):
        os.remove(zipPath)
    file2zip(packagePath, zipPath)
    print("打包完成")


############################## 1. 命令行输入: train训练，predict预测，evaluate评估，pack保存结果 ##############################
if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--mode', choices=['train', 'predict', 'evaluate', 'pack'], required=True)
    # args = parser.parse_args()
    # if args.mode == 'train':
    #     train()
    # elif args.mode == 'predict':
    #     predict()
    # elif args.mode == 'evaluate':
    #     evaluate()
    # elif args.mode == 'pack':
    #     pack()
    train()