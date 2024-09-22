import pandas as pd
import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
# from tiktoken import get_encoding
import re

# .env 파일에서 환경 변수 로드
load_dotenv()

# 스트림릿 앱 시작
st.title("TAG 영화 추천")

# OpenAI 설정
api_key = os.environ.get("LLM_GPT_API_KEY")
base_url = os.environ.get('LLM_GPT_BASE_URL')
model = "openai/gpt-4o-mini-2024-07-18"  

# OpenAI 클라이언트 설정
client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

# 스크립트 파일의 디렉토리 경로를 얻습니다
script_dir = os.path.dirname(os.path.abspath(__file__))
# CSV 파일 읽기
csv_path = os.path.join(script_dir, 'movies.csv')
table_data = pd.read_csv(csv_path)

def generate_query(user_query):
    prompt = f"""
    다음 CSV 파일 구조를 기반으로 사용자의 질문에 대한 pandas DataFrame 쿼리를 생성해주세요:
    
    컬럼: title(제목), genre(장르), year(개봉년도), date(개봉월일), rating(평점), vote_count(평가수), plot(줄거리), main_act(주연), supp_act(조연)
    
    주의사항:
    1. 장르는 '|'로 구분되어 있을 수 있으므로, 장르 검색 시 str.contains() 메소드를 사용하세요.
    2. rating(평점)과 vote_count(평가수)는 숫자 데이터입니다.
    3. year(개봉년도)는 4자리 숫자입니다.
    4. 텍스트 검색은 str.contains() 메소드를 사용하여 부분 일치를 구현하세요.
    5. 쿼리는 pandas의 query() 메소드에서 사용할 수 있는 형식으로 작성해주세요. (예: 'rating >= 9')
    6. 컬럼 이름은 반드시 영어로 사용해야 합니다.
    7. 날짜 범위 검색이 필요한 경우 '@date' 형식을 사용하세요.
    8. 복잡한 조건이 필요한 경우 여러 조건을 '&'로 연결하세요.
    9. 쿼리만 작성하고 추가 설명은 하지 마세요.
    10. 만약 특정 컬럼을 지정할 수 없다면, 모든 텍스트 컬럼에서 검색하도록 쿼리를 작성하세요.
    11. '최근 5년'과 같은 상대적인 시간 표현은 현재 연도를 기준으로 계산하여 구체적인 연도 범위로 변환하세요.
    12. 복합 조건을 정확히 처리하세요. 예를 들어, '평점이 9점 이상인 액션 영화'는 'rating >= 9 & genre.str.contains("액션", case=False)'와 같이 처리해야 합니다.
    13. '추천할 만한', '인기 있는' 등의 표현은 평점(rating)과 평가 수(vote_count)를 기준으로 해석하세요. 예를 들어, 'rating >= 8 & vote_count >= 1000'과 같이 처리할 수 있습니다.
    14. '가장 높은', '최고의' 등의 표현은 무시하고 해당 조건에 맞는 모든 영화를 검색하도록 쿼리를 작성하세요. 결과는 이미 평점 내림차순으로 정렬되어 있습니다.
    
    사용자 질문: {user_query}
    
    쿼리 형식: 
    condition1 & condition2 & ...
    """
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "당신은 pandas DataFrame 쿼리를 생성하는 전문가입니다."},
            {"role": "user", "content": prompt}
        ]
    )
    
    generated_query = response.choices[0].message.content.strip()
    # 따옴표와 백틱 제거
    generated_query = generated_query.strip("'\"```")
    # 언어 식별자 제거 (예: python)
    generated_query = generated_query.split('\n')[-1] if '\n' in generated_query else generated_query
    
    if not generated_query:
        generated_query = "title.str.contains(@user_query, case=False) | plot.str.contains(@user_query, case=False) | main_act.str.contains(@user_query, case=False) | supp_act.str.contains(@user_query, case=False)"
    
    # st.write(f"생성된 쿼리:\n{generated_query}")  # 디버깅을 위해 생된 쿼리 출력
    return generated_query

def query_database(query, data):
    """데이터베이스에 쿼리를 실행하고 결과를 반환합니다."""
    try:
        # 데이터 타입 변환
        data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
        data['vote_count'] = pd.to_numeric(data['vote_count'], errors='coerce')
        data['year'] = pd.to_numeric(data['year'], errors='coerce')
        data['date'] = pd.to_datetime(data['date'], format='%m.%d', errors='coerce')
        
        # 쿼리 실행
        if 'str.contains' in query:
            # str.contains 메소드를 사용하는 경우
            result = data.query(query, local_dict={'user_query': user_query})
        else:
            # 일반적인 쿼리 실행
            result = data.query(query)
        
        # 평점 내림차순으로 정렬하고 상위 5개만 선택
        result = result.sort_values(by='rating', ascending=False).head(5)
        
        return result
    except Exception as e:
        st.error(f"쿼리 실행 중 오류가 발생했습니다: {str(e)}")
        return pd.DataFrame()  # 빈 데이터프레임 반환

def calculate_cost(input_tokens, output_tokens):
    """토큰 수를 기반으로 비용을 계산합니다."""
    input_cost = (input_tokens / 1_000_000) * 0.15
    output_cost = (output_tokens / 1_000_000) * 0.60
    total_cost = input_cost + output_cost
    return total_cost

def generate_response(prompt, context):
    messages = [
        {"role": "system", "content": "당신은 영화 추천 전문가이자 세계 지식을 갖춘 AI입니다. 주어진 정보와 세계 지식을 결합하여 사용자의 질문에 답변해주세요. 응답은 다음 형식을 따라야 합니다:\n\n영화 제목 (개봉 연도) - 장르\n\n평점: X.XX\n간단한 영화 설명 및 추천 이유\n\n관련된 지식 및 흥미로운 사실\n\n각 영화 추천 사이에 구분선을 추가합니다."},
        {"role": "user", "content": f"컨텍스트: {context}\n\n질문: {prompt}\n\n각 영화에 대해 관련된 세계 지식과 흥미로운 사실을 추가해주세요."}
    ]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    
    answer = response.choices[0].message.content.strip()
    total_tokens = response.usage.total_tokens
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost = calculate_cost(input_tokens, output_tokens)
    
    return answer, total_tokens, input_tokens, output_tokens, cost, context

def tag_movie_recommendation(user_query, max_attempts=10):
    for attempt in range(max_attempts):
        generated_query = generate_query(user_query)
        
        filtered_data = query_database(generated_query, table_data)
        
        if len(filtered_data) > 0:
            available_columns = filtered_data.columns.tolist()
            required_columns = ['title', 'genre', 'year', 'rating', 'plot', 'main_act', 'supp_act']
            columns_to_use = [col for col in required_columns if col in available_columns]
            
            filtered_table_data = filtered_data[columns_to_use]
            
            # 세계 지식을 결합한 프롬프트 생성
            context_with_world_knowledge = "영화 정보:\n\n"
            for _, row in filtered_table_data.iterrows():
                context_with_world_knowledge += f"제목: {row['title']}, 개봉년도: {row['year']}, 장르: {row['genre']}, 평점: {row['rating']}\n"
                context_with_world_knowledge += f"줄거리: {row['plot']}\n"
                context_with_world_knowledge += f"주연: {row['main_act']}, 조연: {row['supp_act']}\n\n"
            
            context_with_world_knowledge += "위 영화들에 대한 추천과 함께, 각 영화와 관련된 세계 지식과 흥미로운 사실을 추가해주세요."
            
            response, total_tokens, input_tokens, output_tokens, cost, input_content = generate_response(user_query, context_with_world_knowledge)
            
            movie_recommendations = re.split(r'\n(?=\S+\s+\(\d{4}\)\s+-)', response)
            return movie_recommendations, total_tokens, input_tokens, output_tokens, cost, input_content
        
    st.error(f"죄송합니다. {max_attempts}번의 시도 후에도 '{user_query}'와 관련된 영화를 찾을 수 없습니다.")
    return [f"죄송합니다. {max_attempts}번의 시도 후에도 '{user_query}'와 관련된 영화를 찾을 수 없습니다."], 0, 0, 0, 0, ""

# 스트림릿 인터페이스
predefined_questions = [
    "직접 입력",
    "평점이 9점 이상인 영화 목록을 보여줘",
    "2010년 이후에 개봉한 액션 영화 중 평점이 가장 높은 영화는?",
    "레오나르도 디카프리오가 출연한 영화 중 평점이 가장 높은 영화는?"
]

selected_question = st.selectbox("질문을 선택하거나 직접 입력하세요:", predefined_questions)

if selected_question == "직접 입력":
    user_query = st.text_input("질문을 입력하세요:", "")
else:
    user_query = selected_question

if st.button("추천 받기") and user_query:
    with st.spinner("영화를 검색 중입니다..."):
        recommendations, total_tokens, input_tokens, output_tokens, cost, input_content = tag_movie_recommendation(user_query)
    
    for i, recommendation in enumerate(recommendations):
        st.write(recommendation.strip())
    
    # 데이터프레임 생성
    data = {
        '항목': ['사용된 총 토큰 수', '입력 토큰 수', '출력 토큰 수', '예상 비용'],
        '값': [total_tokens, input_tokens, output_tokens, f"${cost:.6f}"]
    }
    df_results = pd.DataFrame(data)
    
    # 데이터프레임 표시
    st.dataframe(df_results)
