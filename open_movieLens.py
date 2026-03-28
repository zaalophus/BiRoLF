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

if __name__ == "__main__":
    print(find_movies_without_genres())
    # ratings = read_ratings()
    # movies = read_movies()
    # users = read_users()

    # print(ratings.head())
    # print(movies.head())
    # print(users.head())
