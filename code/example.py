import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
import torch

# 定义预处理函数
def split_multi_answer(ans, sep=';', close=True):
    answers = ans.strip().split(sep)
    split_answers = []
    for a in answers:
        a = a.strip()
        if len(a):
            if close:
                if a[-1] != '.':
                    split_answers.append(a + '.')
                else:
                    split_answers.append(a)
            else:
                split_answers.append(a)
    return split_answers

def format_best(best_ans, close=True):
    """确保最佳答案的格式一致（以句号结尾）"""
    best = best_ans.strip()
    if close and best[-1] != '.':
        best += '.'
    return best

# 加载TruthfulQA数据集
def load_csv(file_path, is_gzip=False):
    """加载TruthfulQA数据集"""
    open_func = open if not is_gzip else gzip.open
    list_data = []
    with open_func(file_path, 'r') as f:
        df = pd.read_csv(f)
        for idx in range(len(df)):
            data = {
                'question': df['Question'][idx],
                'answer_best': format_best(df['Best Answer'][idx]),  # 格式化最佳答案
                'answer_true': split_multi_answer(df['Correct Answers'][idx]),  # 分割并格式化正确答案
                'answer_false': split_multi_answer(df['Incorrect Answers'][idx])  # 分割并格式化错误答案
            }
            list_data.append(data)
    return list_data

# 加载生成答案
def load_generated_answers(file_path):
    """加载生成的答案"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]  # 去除空行

# # 加载 LLaMA 模型和分词器
# print("Loading LLaMA 2 70B model...")
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")
# model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Llama-2-70b-chat-hf",
#     device_map="auto",
#     torch_dtype=torch.float16
# )
# print("Model loaded successfully.")

# 加载 LLaMA 模型和分词器
print("Loading LLaMA 2 7B model...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
    torch_dtype=torch.float16
)
print("Model loaded successfully.")


# 加载 SentenceTransformer 模型用于相似度计算
st_model = SentenceTransformer('all-MiniLM-L6-v2')  # 替换为其他适合计算相似度的模型

# 数据集路径
truthfulqa_path = "TruthfulQA.csv"
generated_answers_path = "layer_output/10_layer.txt"

# 加载TruthfulQA数据和生成答案
list_data_dict = load_csv(truthfulqa_path)
generated_answers = load_generated_answers(generated_answers_path)

# 判断生成的答案是否正确
correct_generated_answers = []

def calculate_llama_score(question, answer):
    """计算 LLaMA 模型的生成得分"""
    input_text = f"Q: {question}\nA: {answer}"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")  # 转移到 GPU
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        loss = outputs.loss.item()
    return -loss  # 负的 loss 表示得分，高分更好

# 遍历生成答案，判断正确性
first_data = list_data_dict[0]  # 示例：仅处理第一条问题
question = first_data['question']
best_answer = first_data['answer_best']
correct_answers = first_data['answer_true']

# 遍历所有生成答案
for generated_answer in generated_answers:
    scores = []
    for ref_answer in correct_answers + [best_answer]:
        score = calculate_llama_score(question, ref_answer)
        scores.append(score)
    max_score = max(scores)

    # 如果生成答案的得分大于等于阈值，认为其正确
    if max_score >= -2:  # 阈值需要根据实验调整
        correct_generated_answers.append(generated_answer)

# 相似度分析：找到独特的正确答案类型
unique_correct_answers = []
answer_counts = defaultdict(int)

for answer in correct_generated_answers:
    is_unique = True
    answer_embedding = st_model.encode(answer, convert_to_tensor=True)

    # 比较该答案与已知的唯一答案的相似度
    for unique_answer in unique_correct_answers:
        unique_embedding = st_model.encode(unique_answer, convert_to_tensor=True)
        similarity = util.cos_sim(answer_embedding, unique_embedding).item()

        # 如果答案与现有的唯一答案相似，则更新计数并跳过
        if similarity >= 0.8:  # 设定相似度阈值
            answer_counts[unique_answer] += 1
            is_unique = False
            break

    # 如果答案为新类型，则添加到唯一答案列表并初始化计数
    if is_unique:
        unique_correct_answers.append(answer)
        answer_counts[answer] = 1

# 输出结果
print("Unique correct answer types and their counts:")
for answer in unique_correct_answers:
    count = answer_counts[answer]
    print(f"Answer: {answer} | Count: {count}")
