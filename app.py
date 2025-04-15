from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import os
import logging
import traceback
import sys
from recommender import FestivalRecommender
import uvicorn

# 로깅 설정
logger = logging.getLogger('FestivalRecommender')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

class UserInput(BaseModel):
    userid: str
    searchHistory: List[str]
    favorites: List[str]

app = FastAPI(title="Festival Recommender API")

MODEL_FILE = 'festival_recommender.pkl'

@app.get("/health")
async def health_check():
    logger.debug("헬스 체크 요청")
    return {"status": "ok"}

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
        recs = recommender.recommend(user_data, top_n=5)
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

if __name__ == "__main__":
    logger.info("FastAPI 서버 시작")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)