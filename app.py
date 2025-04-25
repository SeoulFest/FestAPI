from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List
import pandas as pd
import joblib
import os
import logging
import traceback
import sys
from recommender import FestivalRecommender
from scheduler import start_scheduler

# 로깅 설정
logger = logging.getLogger('FestivalRecommender')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(console_handler)
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    logger.addHandler(file_handler)

# 스케줄러 시작 플래그
scheduler_started = False

class UserInput(BaseModel):
    userid: str
    searchHistory: List[str]
    favorites: List[str]

class EventInput(BaseModel):
    eventid: int
    title: str
    category: str
    location: str
    description: str | None = None

    @validator('description', pre=True)
    def handle_null_description(cls, v):
        return "" if v is None else v

app = FastAPI(title="Festival Recommender API")

MODEL_FILE = 'festival_recommender.pkl'
CSV_FILE = 'eventlist.csv'

def add_events_to_csv(events: List[dict]):
    logger.debug("CSV에 이벤트 추가 시작")
    try:
        if not os.path.exists(CSV_FILE):
            logger.info(f"CSV 파일 없음, 새로 생성: {CSV_FILE}")
            df = pd.DataFrame(columns=['eventid', 'title', 'category', 'location', 'description'])
            df.to_csv(CSV_FILE, index=False, encoding='utf-8')
        else:
            df = pd.read_csv(CSV_FILE, encoding='utf-8', encoding_errors='ignore')
        
        new_data = pd.DataFrame(events)
        new_data['description'] = new_data['description'].fillna('')
        df = df[~df['eventid'].isin(new_data['eventid'])]
        df = pd.concat([df, new_data], ignore_index=True)
        df = df.fillna('')
        df.to_csv(CSV_FILE, index=False, encoding='utf-8')
        logger.debug(f"CSV에 {len(events)}개 이벤트 추가 완료, 총 행: {len(df)}")
    except Exception as e:
        logger.error(f"CSV 이벤트 추가 실패: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.on_event("startup")
async def startup_event():
    global scheduler_started
    if not scheduler_started:
        logger.info("FastAPI 서버 시작, 스케줄러 초기화")
        start_scheduler()
        scheduler_started = True
    else:
        logger.debug("스케줄러 이미 시작됨")

@app.get("/health")
async def health_check():
    logger.debug("헬스 체크 요청")
    return {"status": "ok"}

@app.get("/event-sync")
async def event_sync_get():
    logger.warning("GET /event-sync 요청 수신, POST 메서드만 지원")
    raise HTTPException(status_code=405, detail="Method Not Allowed: Use POST for /event-sync")

@app.post("/event-sync")
async def add_events(events: List[EventInput]):
    logger.debug("이벤트 추가 요청 수신")
    try:
        events_dict = [event.dict() for event in events]
        add_events_to_csv(events_dict)
        from model_train import train_model
        train_model(CSV_FILE, MODEL_FILE)
        return {"status": "success", "added_events": len(events_dict)}
    except Exception as e:
        logger.error(f"이벤트 추가 실패: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"이벤트 추가 실패: {str(e)}")

@app.get("/recommned")
async def recommend_misspelled():
    logger.warning("GET /recommned 요청 수신, /recommend로 수정 필요")
    raise HTTPException(status_code=404, detail="Not Found: Did you mean /recommend?")

@app.post("/recommend")
async def recommend_festivals(users: List[UserInput]):
    logger.debug("추천 요청 수신")
    try:
        logger.debug(f"모델 파일 확인: {MODEL_FILE}")
        if not os.path.exists(MODEL_FILE):
            logger.error(f"모델 파일 없음: {MODEL_FILE}")
            raise FileNotFoundError(f"모델 파일 없음: {MODEL_FILE}")
        
        logger.debug(f"모델 로드 시작: {MODEL_FILE}")
        recommender = joblib.load(MODEL_FILE)
        logger.debug("모델 로드 완료")
        
        user_data = [user.dict() for user in users]
        logger.debug(f"사용자 데이터: {user_data}")
        recs = recommender.recommend(user_data)  # top_n 제거
        logger.debug(f"추천 결과: {recs}")
        return recs
    except FileNotFoundError as e:
        logger.error(f"파일 에러: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"파일 에러: {str(e)}")
    except AttributeError as e:
        logger.error(f"모델 로드 에러: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"모델 로드 에러: {str(e)}")
    except Exception as e:
        logger.error(f"추천 실패: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"서버 에러: {str(e)}")