import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------
# 기본 설정
# ----------------------------------------------------------------------
PAGE_TITLE = "탐색"
PAGE_ICON = "🔎"

st.set_page_config(
    page_title=f"{PAGE_TITLE} · 뇌졸중 예측 실습실",
    page_icon=PAGE_ICON,
    layout="wide",
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

COLOR_NO = "#7f9bbd"   # 뇌졸중 없음
COLOR_YES = "#d1495b"  # 뇌졸중 있음
GROUP_COLORS = {"겪지 않음": COLOR_NO, "겪음": COLOR_YES}


# ----------------------------------------------------------------------
# 데이터 불러오기
# ----------------------------------------------------------------------
@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    return pd.read_csv(url, encoding="utf-8")


def rate_by_group(data: pd.DataFrame, column: str, labels: dict) -> pd.DataFrame:
    """어떤 열의 값마다 사람 수와 뇌졸중 비율을 구한다."""
    table = (
        data.groupby(column)["stroke"]
        .agg(사람_수="size", 뇌졸중_인원="sum")
        .reset_index()
    )
    table["뇌졸중 비율(%)"] = table["뇌졸중_인원"] / table["사람_수"] * 100
    table[column] = table[column].map(labels).fillna(table[column].astype(str))
    table = table.rename(columns={column: "구분", "사람_수": "사람 수", "뇌졸중_인원": "뇌졸중 인원"})
    return table


def draw_rate_bar(table: pd.DataFrame, title: str):
    figure = px.bar(
        table,
        x="구분",
        y="뇌졸중 비율(%)",
        text=table["뇌졸중 비율(%)"].map(lambda v: f"{v:.2f}%"),
        title=title,
        color="구분",
        color_discrete_sequence=[COLOR_NO, COLOR_YES],
    )
    figure.update_traces(textposition="outside")
    figure.update_layout(showlegend=False, yaxis_title="뇌졸중 비율(%)", xaxis_title="")
    return figure


# ----------------------------------------------------------------------
# 화면 그리기
# ----------------------------------------------------------------------
st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.caption("데이터를 여러 각도에서 들여다보며 무엇이 뇌졸중과 관련 있는지 찾아봅니다.")

try:
    df = load_data(DATA_URL)
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결을 확인해 주세요.")
    st.caption(f"자세한 내용: {error}")
    st.stop()

df["뇌졸중"] = df["stroke"].map({0: "겪지 않음", 1: "겪음"})

# ---------------------------- 1. 분포 ----------------------------
st.subheader("1. 나이와 평균 혈당은 어떻게 퍼져 있을까?")

left, right = st.columns(2)

with left:
    age_hist = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이 분포",
        color_discrete_sequence=[COLOR_NO],
    )
    age_hist.update_layout(xaxis_title="나이(세)", yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(age_hist, use_container_width=True)

with right:
    glucose_hist = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        color_discrete_sequence=[COLOR_YES],
    )
    glucose_hist.update_layout(xaxis_title="평균 혈당", yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(glucose_hist, use_container_width=True)

st.divider()

# ---------------------------- 2. 두 집단 비교 ----------------------------
st.subheader("2. 뇌졸중을 겪은 사람과 겪지 않은 사람은 어떻게 다를까?")

box_left, box_right = st.columns(2)

with box_left:
    age_box = px.box(
        df,
        x="뇌졸중",
        y="age",
        color="뇌졸중",
        category_orders={"뇌졸중": ["겪지 않음", "겪음"]},
        color_discrete_map=GROUP_COLORS,
        title="나이 비교",
        points=False,
    )
    age_box.update_layout(showlegend=False, xaxis_title="", yaxis_title="나이(세)")
    st.plotly_chart(age_box, use_container_width=True)

with box_right:
    glucose_box = px.box(
        df,
        x="뇌졸중",
        y="avg_glucose_level",
        color="뇌졸중",
        category_orders={"뇌졸중": ["겪지 않음", "겪음"]},
        color_discrete_map=GROUP_COLORS,
        title="평균 혈당 비교",
        points=False,
    )
    glucose_box.update_layout(showlegend=False, xaxis_title="", yaxis_title="평균 혈당")
    st.plotly_chart(glucose_box, use_container_width=True)

mean_table = (
    df.groupby("뇌졸중")[["age", "avg_glucose_level"]]
    .mean()
    .reindex(["겪지 않음", "겪음"])
    .reset_index()
    .rename(columns={"age": "나이 평균(세)", "avg_glucose_level": "평균 혈당의 평균"})
)
mean_table["사람 수"] = (
    df.groupby("뇌졸중").size().reindex(["겪지 않음", "겪음"]).values
)
mean_table = mean_table[["뇌졸중", "사람 수", "나이 평균(세)", "평균 혈당의 평균"]]

st.write("**두 집단의 평균값**")
st.dataframe(
    mean_table.style.format({"나이 평균(세)": "{:.1f}", "평균 혈당의 평균": "{:.1f}"}),
    hide_index=True,
    use_container_width=True,
)

st.divider()

# ---------------------------- 3. 고혈압 · 심장병 ----------------------------
st.subheader("3. 고혈압이나 심장병이 있으면 뇌졸중 비율이 높을까?")

hyper_table = rate_by_group(df, "hypertension", {0: "고혈압 없음", 1: "고혈압 있음"})
heart_table = rate_by_group(df, "heart_disease", {0: "심장병 없음", 1: "심장병 있음"})

bar_left, bar_right = st.columns(2)

with bar_left:
    st.plotly_chart(
        draw_rate_bar(hyper_table, "고혈압에 따른 뇌졸중 비율"),
        use_container_width=True,
    )
    st.dataframe(
        hyper_table.style.format({"뇌졸중 비율(%)": "{:.2f}"}),
        hide_index=True,
        use_container_width=True,
    )

with bar_right:
    st.plotly_chart(
        draw_rate_bar(heart_table, "심장병에 따른 뇌졸중 비율"),
        use_container_width=True,
    )
    st.dataframe(
        heart_table.style.format({"뇌졸중 비율(%)": "{:.2f}"}),
        hide_index=True,
        use_container_width=True,
    )

st.divider()

# ---------------------------- 4. bmi 빈 값 ----------------------------
st.subheader("4. 체질량지수(bmi)가 비어 있는 사람들")

bmi_missing = df[df["bmi"].isna()]
bmi_filled = df[df["bmi"].notna()]

total_people = len(df)
missing_people = len(bmi_missing)

missing_table = pd.DataFrame(
    {
        "구분": ["bmi가 비어 있는 사람", "bmi가 적혀 있는 사람", "전체"],
        "사람 수": [missing_people, len(bmi_filled), total_people],
        "뇌졸중 인원": [
            int(bmi_missing["stroke"].sum()),
            int(bmi_filled["stroke"].sum()),
            int(df["stroke"].sum()),
        ],
    }
)
missing_table["뇌졸중 비율(%)"] = (
    missing_table["뇌졸중 인원"] / missing_table["사람 수"] * 100
)

st.dataframe(
    missing_table.style.format({"뇌졸중 비율(%)": "{:.2f}"}),
    hide_index=True,
    use_container_width=True,
)
st.caption(
    f"전체 {total_people:,}명 가운데 {missing_people:,}명의 bmi가 비어 있습니다."
    " 비어 있는 사람들의 뇌졸중 비율이 전체와 얼마나 다른지 살펴보세요."
)

st.divider()

# ---------------------------- 5. 흡연 상태 ----------------------------
st.subheader("5. 흡연 상태별 사람 수")

smoking_table = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
smoking_table["전체 중 비율(%)"] = smoking_table["사람 수"] / total_people * 100

st.dataframe(
    smoking_table.style.format({"전체 중 비율(%)": "{:.2f}"}),
    hide_index=True,
    use_container_width=True,
)
