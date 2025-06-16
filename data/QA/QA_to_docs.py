import json
import concurrent.futures

# 定义处理每个文件的函数
def process_file(file_name, output_name, news_keys):
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    news_list = []
    for item in data:
        for key in news_keys:
            if key in item and item[key]:
                # 删除换行符，或者替换成空格
                cleaned_news = item[key].replace('\n', '')  # 或者 ' ' 替换为一个空格
                news_list.append(cleaned_news)

    # 写入文件，每个 news 之间用换行分隔
    with open(output_name, 'w', encoding='utf-8') as f:
        f.write('\n'.join(news_list))

    print(f"✅ 已完成：{file_name} -> {output_name}")

# 多线程执行任务
def main():
    tasks = [
        ('1doc_QA.json', 'docs_1.txt', ['news1']),
        ('2docs_QA.json', 'docs_2.txt', ['news1', 'news2']),
        ('3docs_QA.json', 'docs_3.txt', ['news1', 'news2', 'news3']),
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for file_name, output_name, news_keys in tasks:
            futures.append(executor.submit(process_file, file_name, output_name, news_keys))
        
        for future, task in zip(futures, tasks):
            try:
                future.result()
            except Exception as e:
                print(f"❌ 错误处理任务 {task[0]}: {e}")

if __name__ == '__main__':
    main()