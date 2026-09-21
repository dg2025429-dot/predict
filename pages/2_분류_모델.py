import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# ----------------------------------------------------------------------
# 기본 설정
# ----------------------------------------------------------------------
PAGE_TITLE = "분류 모델"
PAGE_ICON = "🌳"

st.set_page_config(
    page_title=f"{PAGE_TITLE} · 뇌졸중 예측 실습실",
    page_icon=PAGE_ICON,
    layout="wide",
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

COLOR_NO = "#7f9bbd"   # 뇌졸중 없음
COLOR_YES = "#d1495b"  # 뇌졸중 있음
REGION_NO = "#d9ead3"  # 트리 영역 - 없음
REGION_YES = "#f4cccc"  # 트리 영역 - 있음

RANDOM_STATE = 42

ALL_FEATURES = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
FEATURE_LABELS = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

LR_NAME = "로지스틱 회귀(확률로 답하는 모델)"
DT_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "기준 모델(입력을 보지 않고 다수결로 답하는 모델)"


# ----------------------------------------------------------------------
# 데이터 불러오기
# ----------------------------------------------------------------------
@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    return pd.read_csv(url, encoding="utf-8")


def split_train_test(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """번호(id) 순으로 정렬한 뒤 열 명씩 묶어, 각 묶음의 앞 세 명을 테스트용으로 고정한다."""
    sorted_df = data.sort_values("id").reset_index(drop=True)
    position_in_group = sorted_df.index % 10
    test_mask = position_in_group < 3
    test_df = sorted_df[test_mask].copy()
    train_df = sorted_df[~test_mask].copy()
    return train_df, test_df


def balance_training_data(train_df: pd.DataFrame, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """뇌졸중이 있는 사람 수에 맞추어 없는 사람 수를 무작위로 골라 크기를 맞춘다."""
    positive = train_df[train_df["stroke"] == 1]
    negative = train_df[train_df["stroke"] == 0]
    n = min(len(positive), len(negative))
    balanced = pd.concat(
        [
            negative.sample(n=n, random_state=random_state),
            positive.sample(n=n, random_state=random_state),
        ]
    )
    return balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)


def build_tree_dot(tree_model: DecisionTreeClassifier, feature_cols: list, feature_labels: dict):
    """의사결정트리를 graphviz DOT 문자열로 바꾼다. 마디마다 인원 수·뇌졸중 인원·비율을 적는다."""
    tree = tree_model.tree_
    lines = [
        "digraph Tree {",
        'graph [fontname="NanumGothic"];',
        'node [shape=box, style="filled, rounded", fontname="NanumGothic", fontsize=12];',
        'edge [fontname="NanumGothic", fontsize=11];',
    ]
    leaf_predictions = []
    used_features = set()

    def recurse(node_id: int):
        n_samples = int(tree.n_node_samples[node_id])
        stroke_count = int(tree.value[node_id][0][1])
        ratio = stroke_count / n_samples * 100 if n_samples else 0.0
        is_leaf = tree.children_left[node_id] == tree.children_right[node_id]

        if is_leaf:
            predicted = int(np.argmax(tree.value[node_id][0]))
            leaf_predictions.append(predicted)
            predicted_label = "있음" if predicted == 1 else "아님"
            fill_color = REGION_YES if predicted == 1 else REGION_NO
            label = (
                f"답: {predicted_label}\\n"
                f"인원 {n_samples}명 · 뇌졸중 {stroke_count}명 ({ratio:.1f}%)"
            )
            lines.append(f'n{node_id} [label="{label}", fillcolor="{fill_color}"];')
        else:
            feature_index = tree.feature[node_id]
            feature_name = feature_cols[feature_index]
            used_features.add(feature_name)
            feature_label = feature_labels[feature_name]
            threshold = tree.threshold[node_id]
            label = (
                f"{feature_label} ≤ {threshold:.1f} ?\\n"
                f"인원 {n_samples}명 · 뇌졸중 {stroke_count}명 ({ratio:.1f}%)"
            )
            lines.append(f'n{node_id} [label="{label}", fillcolor="#ffffff"];')
            left_id = tree.children_left[node_id]
            right_id = tree.children_right[node_id]
            lines.append(f'n{node_id} -> n{left_id} [label="예"];')
            lines.append(f'n{node_id} -> n{right_id} [label="아니요"];')
            recurse(left_id)
            recurse(right_id)

    recurse(0)
    lines.append("}")
    return "\n".join(lines), leaf_predictions, used_features


# ----------------------------------------------------------------------
# 화면 시작
# ----------------------------------------------------------------------
st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.caption("속성을 골라 뇌졸중을 예측하는 두 가지 모델을 만들고 비교해 봅니다.")

try:
    df = load_data(DATA_URL)
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결을 확인해 주세요.")
    st.caption(f"자세한 내용: {error}")
    st.stop()

# ---------------------------- 1. 속성 고르기 ----------------------------
st.subheader("1. 입력으로 사용할 속성 고르기")

options_kr = [FEATURE_LABELS[f] for f in ALL_FEATURES]
default_kr = [FEATURE_LABELS[f] for f in DEFAULT_FEATURES]

selected_kr = st.multiselect(
    "모델이 뇌졸중을 예측할 때 볼 속성을 고르세요.",
    options=options_kr,
    default=default_kr,
)
selected_features = [f for f in ALL_FEATURES if FEATURE_LABELS[f] in selected_kr]

if len(selected_features) < 2:
    st.warning("속성을 두 개 이상 골라야 모델을 만들 수 있습니다.")
    st.stop()

st.caption("고른 속성: " + ", ".join(FEATURE_LABELS[f] for f in selected_features))

# ---------------------------- 2. 데이터 나누고 준비하기 ----------------------------
train_df, test_df = split_train_test(df)

if "bmi" in selected_features:
    median_bmi = train_df["bmi"].median()
    train_df["bmi"] = train_df["bmi"].fillna(median_bmi)
    test_df["bmi"] = test_df["bmi"].fillna(median_bmi)

train_balanced = balance_training_data(train_df)

st.caption(
    f"테스트 데이터 {len(test_df):,}명 · 학습 데이터 {len(train_df):,}명"
    f" (뇌졸중 없음 {int((train_df['stroke'] == 0).sum()):,}명,"
    f" 있음 {int((train_df['stroke'] == 1).sum()):,}명)."
    f" 크기를 맞춘 뒤에는 학습에 {len(train_balanced):,}명"
    f" (없음/있음 각 {int((train_balanced['stroke'] == 1).sum()):,}명)을 사용합니다."
)

X_train_bal = train_balanced[selected_features]
y_train_bal = train_balanced["stroke"]
X_train_raw = train_df[selected_features]
y_train_raw = train_df["stroke"]
X_test = test_df[selected_features]
y_test = test_df["stroke"]

# ---------------------------- 3. 모델 학습 ----------------------------
logistic_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
logistic_model.fit(X_train_bal, y_train_bal)

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
)
tree_model.fit(X_train_bal, y_train_bal)

baseline_model = DummyClassifier(strategy="most_frequent")
baseline_model.fit(X_train_raw, y_train_raw)

train_acc_lr = accuracy_score(y_train_bal, logistic_model.predict(X_train_bal))
test_acc_lr = accuracy_score(y_test, logistic_model.predict(X_test))

train_acc_dt = accuracy_score(y_train_bal, tree_model.predict(X_train_bal))
test_acc_dt = accuracy_score(y_test, tree_model.predict(X_test))

train_acc_base = accuracy_score(y_train_raw, baseline_model.predict(X_train_raw))
test_acc_base = accuracy_score(y_test, baseline_model.predict(X_test))

# ---------------------------- 4. 정확도 카드 ----------------------------
st.subheader("2. 모델 정확도 비교")

card_lr, card_dt, card_base = st.columns(3)


def show_accuracy_card(container, title: str, test_acc: float, train_acc: float):
    container.metric(title, f"{test_acc * 100:.2f}%")
    container.caption(f"훈련 정확도 {train_acc * 100:.2f}% · 테스트 정확도 {test_acc * 100:.2f}%")


show_accuracy_card(card_lr, LR_NAME, test_acc_lr, train_acc_lr)
show_accuracy_card(card_dt, DT_NAME, test_acc_dt, train_acc_dt)
show_accuracy_card(card_base, BASE_NAME, test_acc_base, train_acc_base)

st.divider()

# ---------------------------- 5. 산점도 + 경계선 + 트리 영역 ----------------------------
st.subheader("3. 두 속성으로 보는 경계선과 나무의 영역")

axis_left, axis_right = st.columns(2)

with axis_left:
    x_label = st.selectbox(
        "가로축으로 사용할 속성",
        options=[FEATURE_LABELS[f] for f in selected_features],
        index=0,
        key="x_axis_select",
    )
x_col = [f for f in selected_features if FEATURE_LABELS[f] == x_label][0]

remaining_for_y = [f for f in selected_features if f != x_col]
with axis_right:
    y_label = st.selectbox(
        "세로축으로 사용할 속성",
        options=[FEATURE_LABELS[f] for f in remaining_for_y],
        index=0,
        key="y_axis_select",
    )
y_col = [f for f in remaining_for_y if FEATURE_LABELS[f] == y_label][0]

other_features = [f for f in selected_features if f not in (x_col, y_col)]
fixed_values = {f: float(test_df[f].median()) for f in other_features}

if other_features:
    fixed_text = ", ".join(
        f"{FEATURE_LABELS[f]}은(는) {fixed_values[f]:.1f}" for f in other_features
    )
    st.caption(f"그림에 나오지 않는 속성은 테스트 데이터의 중앙값에 세워 두고 계산했습니다: {fixed_text}.")
else:
    st.caption("선택한 속성이 두 개뿐이라 따로 값을 고정할 속성이 없습니다.")

x_min, x_max = float(test_df[x_col].min()), float(test_df[x_col].max())
y_min, y_max = float(test_df[y_col].min()), float(test_df[y_col].max())

# ----- 의사결정트리 영역 칠하기 -----
grid_steps = 80
x_lin = np.linspace(x_min, x_max, grid_steps)
y_lin = np.linspace(y_min, y_max, grid_steps)
xx, yy = np.meshgrid(x_lin, y_lin)

grid_df = pd.DataFrame({x_col: xx.ravel(), y_col: yy.ravel()})
for f in other_features:
    grid_df[f] = fixed_values[f]
grid_df = grid_df[selected_features]
grid_pred = tree_model.predict(grid_df).reshape(xx.shape)

figure = go.Figure()

figure.add_trace(
    go.Contour(
        x=x_lin,
        y=y_lin,
        z=grid_pred,
        showscale=False,
        colorscale=[[0, REGION_NO], [1, REGION_YES]],
        opacity=0.35,
        contours=dict(start=0, end=1, size=1),
        line=dict(width=0),
        hoverinfo="skip",
        name="의사결정트리 영역",
    )
)

# ----- 테스트 데이터 점 찍기 -----
for label, color, value in [("뇌졸중 없음", COLOR_NO, 0), ("뇌졸중 있음", COLOR_YES, 1)]:
    subset = test_df[test_df["stroke"] == value]
    figure.add_trace(
        go.Scatter(
            x=subset[x_col],
            y=subset[y_col],
            mode="markers",
            marker=dict(color=color, size=7, line=dict(width=0.5, color="white")),
            name=label,
        )
    )

# ----- 로지스틱 회귀 경계선(0.5) -----
coefs = logistic_model.coef_[0]
intercept = logistic_model.intercept_[0]
idx_x = selected_features.index(x_col)
idx_y = selected_features.index(y_col)
coef_x = coefs[idx_x]
coef_y = coefs[idx_y]
sum_fixed = intercept + sum(
    coefs[selected_features.index(f)] * fixed_values[f] for f in other_features
)

boundary_note = ""
if abs(coef_y) > 1e-9:
    line_x = np.linspace(x_min, x_max, 200)
    line_y = -(sum_fixed + coef_x * line_x) / coef_y
    inside = (line_y >= y_min) & (line_y <= y_max)
    figure.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            line=dict(color="black", width=2, dash="dash"),
            name="로지스틱 회귀 경계선(0.5)",
        )
    )
    if not inside.any():
        boundary_note = "로지스틱 회귀의 경계선은 이 그림의 범위 밖에 있어 보이지 않습니다."
elif abs(coef_x) > 1e-9:
    x_value = -sum_fixed / coef_x
    figure.add_trace(
        go.Scatter(
            x=[x_value, x_value],
            y=[y_min, y_max],
            mode="lines",
            line=dict(color="black", width=2, dash="dash"),
            name="로지스틱 회귀 경계선(0.5)",
        )
    )
    if x_value < x_min or x_value > x_max:
        boundary_note = "로지스틱 회귀의 경계선은 이 그림의 범위 밖에 있어 보이지 않습니다."
else:
    boundary_note = "이 두 속성만으로는 로지스틱 회귀의 경계선을 그릴 수 없습니다."

figure.update_layout(
    title=f"{FEATURE_LABELS[x_col]} vs {FEATURE_LABELS[y_col]}",
    xaxis_title=FEATURE_LABELS[x_col],
    yaxis_title=FEATURE_LABELS[y_col],
    xaxis=dict(range=[x_min, x_max]),
    yaxis=dict(range=[y_min, y_max]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)

st.plotly_chart(figure, use_container_width=True)

if boundary_note:
    st.caption(boundary_note)

st.divider()

# ---------------------------- 6. 의사결정트리 가지 그림 ----------------------------
st.subheader("4. 의사결정트리가 던진 질문")

dot_text, leaf_predictions, used_features = build_tree_dot(
    tree_model, selected_features, FEATURE_LABELS
)
st.graphviz_chart(dot_text)

total_leaves = len(leaf_predictions)
no_stroke_leaves = leaf_predictions.count(0)

st.write(f"- 답을 내는 마디(잎)는 모두 {total_leaves}칸이고, 그중 {no_stroke_leaves}칸이 '아님'이라고 답합니다.")

if used_features:
    st.write("- 고른 속성 가운데 이 나무가 실제로 물은 것:")
    for feature_name in selected_features:
        if feature_name in used_features:
            st.write(f"  - {FEATURE_LABELS[feature_name]}")
else:
    st.write("- 이 나무는 어떤 속성도 질문에 사용하지 않았습니다.")
