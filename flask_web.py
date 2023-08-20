from pandas import read_csv
from flask import Flask, render_template, request, jsonify
from main_impute import impute_data
import time
import torch

app = Flask(__name__)

# 保存进度信息的全局对象类
class Globel_args():
    def __init__(self):

        self.progress = 0
    
    def get_progress(self):
        return self.progress

    def set_progress(self, value):
        self.progress = value

# args中的progress保存了填充进度信息
args = Globel_args()

# 定义首页路由
@app.route('/')
def index():
    return render_template('index.html')

# 定义数据填充路由
@app.route('/impute', methods=['POST'])
def impute():

    # 获取参数
    data_path = request.form['data_path']
    save_path = request.form['save_path']
    epoch = int(request.form['epoch'])
    n_critic = int(request.form['n_critic'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建一个新的进程来调用数据填充函数
    # impute_thread = Thread(target=impute_data, args=('cuda', 1, epoch, data_path, save_path, args))
    # impute_thread.start()
    impute_data(device, n_critic, epoch, data_path, save_path, args)

    # 跳转到结果页面，并将结果传递给模板
    time.sleep(1)
    result_msg = "填充完毕，已保存在 # " + save_path + " #！"
    # 展示5×5的数据
    data_df = read_csv(save_path)
    data = data_df.iloc[:5, :5]
    return render_template('result.html', result=result_msg, table=data.to_html())

    
# 定义结果显示路由
@app.route('/result')
def result():

    # 跳转到结果页面，并将结果传递给模板
    save_path = 'data/save_data.csv'
    result_msg = "填充完毕，已保存在 # " + save_path + " #！"
    # 展示5×5的数据
    data_df = read_csv(save_path)
    data = data_df.iloc[:5, :5]
    return render_template('result.html', result=result_msg, table=data.to_html())

# 定义获取当前进度的路由
@app.route('/progress')
def get_progress():
    # 将当前进度按JSON对象返回
    return jsonify({'progress': args.get_progress()})


if __name__ == '__main__':
    app.run(debug=True)