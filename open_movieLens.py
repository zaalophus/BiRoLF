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


if __name__ == "__main__":
    ratings = read_ratings()
    movies = read_movies()
    users = read_users()

    print(ratings.head())
    print(movies.head())
    print(users.head())
