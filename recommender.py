import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger('FestivalRecommender')

class FestivalRecommender:
    def __init__(self):
        logger.debug("FestivalRecommender 초기화")
        self.vectorizer = TfidfVectorizer(stop_words=None, lowercase=True)
        self.festival_vectors = None
        self.festival_ids = None
        self.festival_data = None

    def fit(self, festival_data):
        logger.debug("모델 학습 시작")
        try:
            # status가 END가 아닌 데이터만 사용
            if 'status' in festival_data.columns:
                initial_count = len(festival_data)
                festival_data = festival_data[festival_data['status'] != 'END'].copy()
                logger.debug(f"status=END 제외 후 데이터 크기: {len(festival_data)} (제외된 행: {initial_count - len(festival_data)})")
            else:
                logger.warning("status 컬럼이 없음, 모든 데이터 사용")

            self.festival_data = festival_data
            self.festival_ids = festival_data['eventid'].astype(str).values
            logger.debug(f"축제 데이터 크기: {festival_data.shape}")
            festival_data['combined_features'] = (
                festival_data['title'].fillna('') + ' ' +
                festival_data['category'].fillna('') + ' ' +
                festival_data['location'].fillna('') + ' ' +
                festival_data['description'].fillna('')
            )
            logger.debug("combined_features 생성 완료")
            self.festival_vectors = self.vectorizer.fit_transform(festival_data['combined_features'])
            logger.debug("TF-IDF 벡터화 완료")
        except Exception as e:
            logger.error(f"모델 학습 실패: {str(e)}")
            raise

    def recommend(self, user_data, top_n=5):
        logger.debug("추천 생성 시작")
        recommendations = []
        for user in user_data:
            logger.debug(f"사용자 처리: {user['userid']}")
            try:
                if not user['searchHistory'] and not user['favorites']:
                    logger.debug(f"사용자 {user['userid']} 데이터 비어있음")
                    recommendations.append({
                        'userid': user['userid'],
                        'festivalRecommendations': [{'eventid': None}]
                    })
                    continue
                
                user_text = ' '.join(user['searchHistory'] + user['favorites']).lower()
                logger.debug(f"사용자 텍스트: {user_text}")
                user_vector = self.vectorizer.transform([user_text])
                logger.debug("사용자 벡터 생성")
                similarities = cosine_similarity(user_vector, self.festival_vectors).flatten()
                logger.debug("유사도 계산 완료")
                top_indices = np.argsort(similarities)[-top_n:][::-1]
                recommended_ids = self.festival_ids[top_indices].tolist()
                recommendations.append({
                    'userid': user['userid'],
                    'festivalRecommendations': [{'eventid': recommended_ids}]
                })
            except Exception as e:
                logger.error(f"사용자 {user['userid']} 추천 실패: {str(e)}")
                recommendations.append({
                    'userid': user['userid'],
                    'festivalRecommendations': [{'eventid': None}]
                })
        logger.debug("추천 생성 완료")
        return recommendations