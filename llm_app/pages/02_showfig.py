import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@st.cache_data
def read_data(filepath):
    """
    キャッシュをすることでリロードするたびにロードされるのを防ぐ関数
    """
    return pd.read_csv(filepath)


#resultが追加された場合は、resultsとnamesにも追記

st.title('Evaluation Using :blue[RAGAS] Framework')




# Difference in results by random sampling-----------------------------------------------------------
container1 = st.container(border=True)
container1.subheader("Difference in results by random sampling", divider='blue')

#ランダムサンプリングによる差を見るためのデータ
# resultのデータ
result = {'faithfulness': 0.8667, 'answer_relevancy': 0.9210, 'context_precision': 1.0000, 'answer_correctness': 0.7930, 'context_recall': 1.0000}
# result2のデータ
result2 = {'faithfulness': 1.0000, 'answer_relevancy': 0.9182, 'context_precision': 1.0000, 'answer_correctness': 0.8502, 'context_recall': 1.0000}
# result3のデータ
result3 = {'faithfulness': 0.8429, 'answer_relevancy': 0.9191, 'context_precision': 1.0000, 'answer_correctness': 0.7685, 'context_recall': 1.0000}


# グラフ作成
fig = go.Figure()

# データをリスト化
results = [result, result2, result3]
names = ['Random Sample 1', 'Random Sample 2', 'Random Sample 3']
#colors = ['blue', 'orange', 'green']  # データごとの色
colors = ['rgba(0, 0, 255, 0.3)', 'rgba(255, 165, 0, 0.3)', 'rgba(0, 128, 0, 0.3)']  # データごとの色（透明度を設定）

# データを追加
for i, res in enumerate(results):
    data = {
        'context_precision': res['context_precision'],
        'faithfulness': res['faithfulness'],
        'answer_relevancy': res['answer_relevancy'],
        'context_recall': res['context_recall'],
        'answer_correctness': res['answer_correctness'],
    }
    fig.add_trace(go.Scatterpolar(
        r=list(data.values()),
        theta=list(data.keys()),
        fill='toself',
        name=names[i],  # レジェンドの名前
        #line=dict(color=colors[i])  # グラフの色を指定
    ))

# レイアウト設定
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
    showlegend=True,
    title='Retrieval Augmented Generation - Evaluation',
    width=800,
)



# 棒グラフ用のデータ
# categories = ['faithfulness', 'answer_relevancy', 'answer_correctness']
categories = list(result.keys())

# 棒グラフ用のデータを生成
values = [[res[cat] for res in results] for cat in categories]

# 棒グラフ作成
bar_fig = go.Figure()

# 全てのカテゴリーの棒グラフを追加
for i, name in enumerate(names):
    bar_fig.add_trace(go.Bar(
        x=categories,
        y=[val[i] for val in values],
        name=name,  # レジェンドの名前
        #marker=dict(color=colors[i])  # 棒グラフの色を指定
    ))

# レイアウト設定
bar_fig.update_layout(
    title='Retrieval Augmented Generation - Evaluation (Bar Chart)',
    xaxis=dict(title='Result'),  # x軸ラベル
    yaxis=dict(title='Value', range=[0, 1]),  # y軸ラベルと値の範囲
    showlegend=True,
    #barmode='group',
    bargroupgap=0.1,
    width=1200,
    #height = 500,
)

# グラフを表示
container1.plotly_chart(fig)
container1.write("")
container1.plotly_chart(bar_fig)




#Difference in results by sample size----------------------------------------------------------------------------------------------------------------------------------
container2 = st.container(border=True)

container2.subheader("Difference in results by sample size", divider='blue')

result_size_5 = {'faithfulness': 0.9000, 'answer_relevancy': 0.9222, 'context_precision': 1.0000, 'answer_correctness': 0.7124, 'context_recall': 1.0000}
result_size_10 = {'faithfulness': 1.0000, 'answer_relevancy': 0.8335, 'context_precision': 0.9000, 'answer_correctness': 0.6865, 'context_recall': 0.9000}
result_size_15 = {'faithfulness': 0.9022, 'answer_relevancy': 0.8714, 'context_precision': 0.9333, 'answer_correctness': 0.7082, 'context_recall': 0.9333}

# グラフ作成
fig = go.Figure()

# データをリスト化
results = [result_size_5, result_size_10, result_size_15]
names = ['Sample Size 5', 'Sample Size 10', 'Sample Size 15']
#colors = ['blue', 'orange', 'green']  # データごとの色
colors = ['rgba(0, 0, 255, 0.3)', 'rgba(255, 165, 0, 0.3)', 'rgba(0, 128, 0, 0.3)']  # データごとの色（透明度を設定）

# データを追加
for i, res in enumerate(results):
    data = {
        'context_precision': res['context_precision'],
        'faithfulness': res['faithfulness'],
        'answer_relevancy': res['answer_relevancy'],
        'context_recall': res['context_recall'],
        'answer_correctness': res['answer_correctness'],
    }
    fig.add_trace(go.Scatterpolar(
        r=list(data.values()),
        theta=list(data.keys()),
        fill='toself',
        name=names[i],  # レジェンドの名前
        #line=dict(color=colors[i])  # グラフの色を指定
    ))

# レイアウト設定
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
    showlegend=True,
    title='Retrieval Augmented Generation - Evaluation',
    width=800,
)



# 棒グラフ用のデータ
# categories = ['faithfulness', 'answer_relevancy', 'answer_correctness']
categories = list(result.keys())

# 棒グラフ用のデータを生成
values = [[res[cat] for res in results] for cat in categories]

# 棒グラフ作成
bar_fig = go.Figure()

# 全てのカテゴリーの棒グラフを追加
for i, name in enumerate(names):
    bar_fig.add_trace(go.Bar(
        x=categories,
        y=[val[i] for val in values],
        name=name,  # レジェンドの名前
        #marker=dict(color=colors[i])  # 棒グラフの色を指定
    ))

# レイアウト設定
bar_fig.update_layout(
    title='Retrieval Augmented Generation - Evaluation (Bar Chart)',
    xaxis=dict(title='Result'),  # x軸ラベル
    yaxis=dict(title='Value', range=[0, 1]),  # y軸ラベルと値の範囲
    showlegend=True,
    #barmode='group',
    bargroupgap=0.1,
    width=1200,
    #height = 500,
)

# グラフを表示
container2.plotly_chart(fig)
container2.write("")
container2.plotly_chart(bar_fig)





#Score per question-----------------------------------------------------------------
container3 = st.container(border=True)

container3.subheader("Score per question", divider='blue')
# boxplot
import plotly.express as px
df = px.data.tips()
fig = px.box(df, x="time", y="total_bill")
fig.show()


df = read_data("../result_df.csv")

# ワイド形式からロング形式に変換
long_df = pd.melt(df, id_vars=['question', 'ground_truth', 'answer', 'contexts'],
                  value_vars=['faithfulness', 'answer_relevancy', 'context_precision', 'answer_correctness', 'context_recall'],
                  var_name='metrics', value_name='value')

fig_boxplot = px.box(long_df, x="metrics", y="value", points="all", hover_data=["question", "answer"])
container3.plotly_chart(fig_boxplot)





#RAG有無での評価比較---------------------------------------------------------------------------------------------------------------------------------------------
container4 = st.container(border=True)
container4.subheader("Differences in results with and without RAG", divider='blue')


result_no_rag = {'answer_correctness': 0.3709}
result_rag = {'answer_correctness': 0.6154}

# データをリスト化
results = [result_no_rag, result_rag]
names = ['without RAG', 'with RAG 2']


# 棒グラフ用のデータ
# categories = ['faithfulness', 'answer_relevancy', 'answer_correctness']
categories = list(result_rag.keys())

# 棒グラフ用のデータを生成
values = [[res[cat] for res in results] for cat in categories]

# 棒グラフ作成
bar_fig = go.Figure()

# 全てのカテゴリーの棒グラフを追加
for i, name in enumerate(names):
    bar_fig.add_trace(go.Bar(
        x=categories,
        y=[val[i] for val in values],
        name=name,  # レジェンドの名前
        #marker=dict(color=colors[i])  # 棒グラフの色を指定
    ))

# レイアウト設定
bar_fig.update_layout(
    title='Retrieval Augmented Generation - Evaluation (Bar Chart)',
    xaxis=dict(title='Result'),  # x軸ラベル
    yaxis=dict(title='Value', range=[0, 1]),  # y軸ラベルと値の範囲
    showlegend=True,
    #barmode='group',
    bargroupgap=0.1,
    width=1200,
    #height = 500,
)

# グラフを表示
container4.plotly_chart(bar_fig)
