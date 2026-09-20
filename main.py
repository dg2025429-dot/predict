import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# 기본 설정 (브라우저 탭 제목 + 아이콘)
# ----------------------------------------------------------------------
APP_TITLE = "뇌졸중 예측 실습실"
APP_ICON = "🧠"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"


# ----------------------------------------------------------------------
# 데이터 불러오기
# ----------------------------------------------------------------------
@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    return pd.read_csv(url, encoding="utf-8")


def describe_kind(series: pd.Series) -> str:
    """열의 값이 어떤 종류인지 우리말로 설명한다."""
    n_unique = series.nunique(dropna=True)

    if pd.api.types.is_numeric_dtype(series):
        if n_unique <= 5:
            values = sorted(series.dropna().unique().tolist())
            values_text = ", ".join(str(v) for v in values)
            return f"숫자 (값 {n_unique}가지: {values_text})"
        return f"숫자 (서로 다른 값 {n_unique}개)"

    if n_unique <= 8:
        values = sorted(series.dropna().astype(str).unique().tolist())
        values_text = ", ".join(values)
        return f"글자 (값 {n_unique}가지: {values_text})"
    return f"글자 (서로 다른 값 {n_unique}개)"


# ----------------------------------------------------------------------
# 화면 그리기
# ----------------------------------------------------------------------
st.title(f"{APP_ICON} {APP_TITLE}")
st.caption("첫 번째 화면 · 우리가 다룰 데이터가 무엇인지 알아봅니다.")

try:
    df = load_data(DATA_URL)
except Exception as error:  # 인터넷 연결이 막혔을 때 등
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결을 확인해 주세요.")
    st.caption(f"자세한 내용: {error}")
    st.stop()

# ---------------------------- 큰 숫자 카드 ----------------------------
st.subheader("한눈에 보기")

total_people = len(df)
total_columns = df.shape[1]
stroke_people = int((df["stroke"] == 1).sum())
stroke_ratio = stroke_people / total_people * 100 if total_people else 0

card1, card2, card3, card4 = st.columns(4)
card1.metric("전체 사람 수", f"{total_people:,}명")
card2.metric("열 개수", f"{total_columns}개")
card3.metric("뇌졸중을 겪은 사람 수", f"{stroke_people:,}명")
card4.metric("뇌졸중을 겪은 사람 비율", f"{stroke_ratio:.2f}%")

st.divider()

# ---------------------------- 열 설명 표 ----------------------------
st.subheader("열 살펴보기")
st.write("**우리말 뜻** 칸은 비어 있습니다. 교재를 보고 직접 채워 넣어 보세요.")

column_table = pd.DataFrame(
    {
        "열 이름": df.columns,
        "우리말 뜻": ["" for _ in df.columns],
        "값의 종류": [describe_kind(df[col]) for col in df.columns],
        "빈 값 개수": [int(df[col].isna().sum()) for col in df.columns],
    }
)

edited_table = st.data_editor(
    column_table,
    hide_index=True,
    use_container_width=True,
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름", width="small"),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            help="교재를 보고 직접 적어 보세요.",
            width="medium",
        ),
        "값의 종류": st.column_config.TextColumn("값의 종류", width="large"),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", width="small"),
    },
    key="column_meaning_table",
)

filled_count = int((edited_table["우리말 뜻"].astype(str).str.strip() != "").sum())
st.caption(f"채운 칸: {filled_count} / {len(edited_table)}")

st.divider()

# ---------------------------- 데이터 맛보기 ----------------------------
st.subheader("데이터 맛보기 (처음 다섯 줄)")
st.dataframe(df.head(5), use_container_width=True)

st.divider()

# ---------------------------- 데이터 출처 ----------------------------
st.subheader("데이터 출처")
st.write("교재에 있는 출처를 아래 칸에 적어 보세요.")

source_text = st.text_area(
    "출처 적는 곳",
    value="",
    height=120,
    placeholder="여기에 교재에 있는 데이터 출처를 적습니다.",
    key="data_source",
)

if source_text.strip():
    st.success("출처를 적었습니다.")
    st.markdown(f"> {source_text.strip()}")
else:
    st.info("아직 비어 있습니다.")
