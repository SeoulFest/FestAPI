import pandas as pd
import joblib
import logging
import sys
import traceback
import os
from recommender import FestivalRecommender

# 로깅 설정
logger = logging.getLogger('FestivalRecommender')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)
file_handler = logging.FileHandler('model_train.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

def load_festival_data(file_path):
    logger.debug(f"축제 데이터 로드: {file_path}")
    try:
        if not os.path.exists(file_path):
            logger.error(f"CSV 파일 없음: {file_path}")
            raise FileNotFoundError(f"CSV 파일 없음: {file_path}")
        df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='ignore')
        logger.debug(f"데이터프레임 크기: {df.shape}")
        required_columns = ['eventid', 'title', 'category', 'location', 'description']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"컬럼 {col} 누락, 빈 문자열로 채움")
                df[col] = ''
        if 'status' in df.columns:
            logger.debug(f"status 컬럼 존재, 값 분포: {df['status'].value_counts().to_dict()}")
        else:
            logger.warning("status 컬럼이 없음")
        return df
    except Exception as e:
        logger.error(f"데이터 로드 실패: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    logger.info("모델 학습 스크립트 시작")
    festival_file = 'eventlist.csv'
    try:
        festival_data = load_festival_data(festival_file)
        recommender = FestivalRecommender()
        recommender.fit(festival_data)
        model_file = 'festival_recommender.pkl'
        joblib.dump(recommender, model_file)
        logger.info(f"모델이 {model_file}에 저장되었습니다.")
    except Exception as e:
        logger.error(f"모델 생성 실패: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)