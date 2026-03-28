import re
import pandas as pd


def read_movies() -> pd.DataFrame:
    movies = pd.read_csv(
        "3. Data/MovieLens/movies.dat",
        sep="::",
        engine="python",
        names=["MovieID", "Title", "Genres"],
        encoding="latin-1"
    )

    # Title 끝의 "(year)" 패턴 추출 — 제목 자체에 ()가 있을 수 있으므로 마지막 것만 추출
    pattern = re.compile(r'^(.*)\((\d{4})\)\s*$')

    def split_title_year(title):
        m = pattern.match(title.strip())
        if m:
            return m.group(1).strip(), int(m.group(2))
        return title.strip(), None

    movies[["Title", "Year"]] = movies["Title"].apply(
        lambda t: pd.Series(split_title_year(t))
    )

    return movies[["MovieID", "Title", "Year", "Genres"]]


def read_ratings() -> pd.DataFrame:
    ratings = pd.read_csv(
        "3. Data/MovieLens/ratings.dat",
        sep="::",
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"]
    )
    return ratings


def read_users(with_unobservable: bool = False, unobservable_feature: str = "Gender") -> pd.DataFrame:
    users = pd.read_csv(
        "3. Data/MovieLens/users.dat",
        sep="::",
        engine="python",
        names=["UserID", "Gender", "Age", "Occupation", "Zip"]
    )
    if with_unobservable:
        users = users.drop(columns=[unobservable_feature])
    return users

def find_movies_without_genres(movies: pd.DataFrame = None) -> pd.DataFrame:
    """
    주어진 movies DataFrame에서 장르가 없는 영화를 반환한다.

    MovieLens 데이터에서 장르가 없는 경우는 두 가지:
      - Genres 컬럼이 NaN / 빈 문자열
      - Genres 값이 "(no genres listed)" (MovieLens 공식 표기)

    Parameters
    ----------
    movies : pd.DataFrame or None
        read_movies()로 읽은 DataFrame. None이면 내부에서 직접 로드한다.

    Returns
    -------
    pd.DataFrame
        장르가 없는 영화 행만 포함한 DataFrame.
    """
    if movies is None:
        movies = read_movies()

    no_genre_mask = (
        movies["Genres"].isna()
        | (movies["Genres"].str.strip() == "")
        | (movies["Genres"].str.strip().str.lower() == "(no genres listed)")
    )
    print(movies["Genres"].unique())
    return movies[no_genre_mask].reset_index(drop=True)

def check_embeddings(n_movies: int = 10, model_name: str = "BAAI/bge-base-en-v1.5"):
    """
    n_movies개의 영화를 샘플하여 제목/장르 임베딩이 제대로 동작하는지 확인한다.

    출력 내용:
    - 임베딩 shape
    - 임베딩 벡터 norm (정상이면 0이 아닌 값)
    - 유사도 행렬 (title 기준): 같은 장르 영화끼리 높아야 자연스러움
    - 제목/장르별 최근접 이웃 (top-3)
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer

    movies = read_movies()
    sample = movies.sample(n=min(n_movies, len(movies)), random_state=42).reset_index(drop=True)

    model = SentenceTransformer(model_name)

    # Title embedding
    title_emb = model.encode(sample['Title'].tolist(), convert_to_numpy=True)  # (M, dim)

    # Genre embedding: mean-pool per movie
    genre_emb = np.stack([
        model.encode(row.split('|'), convert_to_numpy=True).mean(axis=0)
        for row in sample['Genres']
    ])  # (M, dim)

    print(f"=== Embedding check (model: {model_name}) ===")
    print(f"Title emb shape : {title_emb.shape}")
    print(f"Genre emb shape : {genre_emb.shape}")
    print(f"Title emb norms : {np.linalg.norm(title_emb, axis=1).round(4)}")
    print(f"Genre emb norms : {np.linalg.norm(genre_emb, axis=1).round(4)}")

    # Cosine similarity matrix (title)
    t_norm = title_emb / np.linalg.norm(title_emb, axis=1, keepdims=True)
    sim_title = t_norm @ t_norm.T

    g_norm = genre_emb / np.linalg.norm(genre_emb, axis=1, keepdims=True)
    sim_genre = g_norm @ g_norm.T

    print("\n=== Top-3 nearest neighbors (by title embedding) ===")
    for i, row in sample.iterrows():
        sims  = sim_title[i].copy()
        sims[i] = -1  # exclude self
        top3  = np.argsort(sims)[::-1][:3]
        neighbors = [f"{sample.loc[j,'Title']} ({sims[j]:.3f})" for j in top3]
        print(f"  [{row['Title']} | {row['Genres']}]")
        print(f"    → {', '.join(neighbors)}")

    print("\n=== Top-3 nearest neighbors (by genre embedding) ===")
    for i, row in sample.iterrows():
        sims  = sim_genre[i].copy()
        sims[i] = -1
        top3  = np.argsort(sims)[::-1][:3]
        neighbors = [f"{sample.loc[j,'Title']} [{sample.loc[j,'Genres']}] ({sims[j]:.3f})" for j in top3]
        print(f"  [{row['Title']} | {row['Genres']}]")
        print(f"    → {', '.join(neighbors)}")


if __name__ == "__main__":
    check_embeddings(n_movies=10)
    # ratings = read_ratings()
    # movies = read_movies()
    # users = read_users()

    # print(ratings.head())
    # print(movies.head())
    # print(users.head())
