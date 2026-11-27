"""Generate SYNTHETIC data for the Online Course Engagement project.

Creates three related tables and saves them as CSV files in data/:
    - learners.csv     (2000 rows)
    - courses.csv      (20 rows)
    - enrollments.csv  (4500 rows)

The data is entirely artificial. A hidden "motivation" score per learner,
together with course and learner attributes, drives the probability of
completing a course; the observed engagement metrics (sessions, lessons,
quiz scores, ...) are generated from the same hidden factors plus noise.

Run from the project root:
    python src/data_generator.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

SEED = 42
N_LEARNERS = 2000
N_ENROLLMENTS = 4500

SIGNUP_START = pd.Timestamp("2023-01-01")
SIGNUP_END = pd.Timestamp("2024-12-31")
MAX_ENROLL_DELAY_DAYS = 90  # enrollment happens at most 90 days after signup

# (title, category, difficulty) for the 20 courses
COURSE_CATALOG = [
    ("Python Basics", "Programming", "Beginner"),
    ("Web Development Foundations", "Programming", "Beginner"),
    ("Data Structures in Practice", "Programming", "Intermediate"),
    ("Advanced Software Design", "Programming", "Advanced"),
    ("Statistics for Beginners", "Data Science", "Beginner"),
    ("Data Analysis with Pandas", "Data Science", "Intermediate"),
    ("Machine Learning Essentials", "Data Science", "Intermediate"),
    ("Deep Learning Workshop", "Data Science", "Advanced"),
    ("Business Fundamentals", "Business", "Beginner"),
    ("Project Management", "Business", "Intermediate"),
    ("Financial Modeling", "Business", "Advanced"),
    ("Product Strategy", "Business", "Intermediate"),
    ("Design Thinking", "Design", "Beginner"),
    ("UX Research Methods", "Design", "Intermediate"),
    ("Advanced Interface Design", "Design", "Advanced"),
    ("Digital Marketing Basics", "Marketing", "Beginner"),
    ("Content Marketing", "Marketing", "Intermediate"),
    ("Marketing Analytics", "Marketing", "Advanced"),
    ("SEO Fundamentals", "Marketing", "Beginner"),
    ("Brand Management", "Marketing", "Intermediate"),
]

# Course length ranges by difficulty: (weeks_min, weeks_max, lessons_min, lessons_max)
COURSE_SIZE = {
    "Beginner": (4, 6, 10, 25),
    "Intermediate": (5, 8, 20, 40),
    "Advanced": (8, 12, 30, 60),
}

# Effect of difficulty on the log-odds of completing a course
DIFFICULTY_EFFECT = {"Beginner": 0.4, "Intermediate": 0.0, "Advanced": -0.5}

COUNTRIES = ["United States", "India", "United Kingdom", "Germany",
             "Brazil", "Canada", "Nigeria", "Poland"]
COUNTRY_WEIGHTS = [0.25, 0.22, 0.10, 0.09, 0.10, 0.08, 0.09, 0.07]

DEVICES = ["Mobile", "Desktop", "Tablet"]
DEVICE_WEIGHTS = [0.45, 0.45, 0.10]
DEVICE_EFFECT = {"Mobile": -0.25, "Desktop": 0.15, "Tablet": 0.0}

CHANNELS = ["Organic Search", "Paid Ads", "Referral", "Social Media", "Email"]
CHANNEL_WEIGHTS = [0.30, 0.22, 0.15, 0.20, 0.13]
# Shift of the hidden motivation score by acquisition channel
CHANNEL_MOTIVATION = {"Organic Search": 0.1, "Paid Ads": -0.25, "Referral": 0.3,
                      "Social Media": -0.15, "Email": 0.15}

EXPERIENCE_LEVELS = ["No experience", "Some experience", "Experienced"]
EXPERIENCE_WEIGHTS = [0.35, 0.45, 0.20]
EXPERIENCE_EFFECT = {"No experience": -0.2, "Some experience": 0.0, "Experienced": 0.25}


def generate_courses(rng: np.random.Generator) -> pd.DataFrame:
    """Create the course catalog table."""
    rows = []
    for course_id, (title, category, difficulty) in enumerate(COURSE_CATALOG, start=1):
        w_min, w_max, l_min, l_max = COURSE_SIZE[difficulty]
        is_free = int(rng.random() < 0.25)
        price = 0 if is_free else int(rng.choice([29, 49, 79, 99, 149]))
        rows.append({
            "course_id": course_id,
            "title": title,
            "category": category,
            "difficulty": difficulty,
            "duration_weeks": int(rng.integers(w_min, w_max + 1)),
            "n_lessons": int(rng.integers(l_min, l_max + 1)),
            "is_free": is_free,
            "price_usd": price,
        })
    return pd.DataFrame(rows)


def generate_learners(rng: np.random.Generator):
    """Create the learners table and the hidden motivation score per learner."""
    n = N_LEARNERS
    span_days = (SIGNUP_END - SIGNUP_START).days
    signup_dates = SIGNUP_START + pd.to_timedelta(rng.integers(0, span_days + 1, n), unit="D")

    channel = rng.choice(CHANNELS, size=n, p=CHANNEL_WEIGHTS)
    learners = pd.DataFrame({
        "learner_id": np.arange(1, n + 1),
        "signup_date": signup_dates.strftime("%Y-%m-%d"),
        "country": rng.choice(COUNTRIES, size=n, p=COUNTRY_WEIGHTS),
        "age": np.clip(rng.normal(32, 9, n), 18, 65).round().astype(int),
        "device": rng.choice(DEVICES, size=n, p=DEVICE_WEIGHTS),
        "acquisition_channel": channel,
        "prior_experience": rng.choice(EXPERIENCE_LEVELS, size=n, p=EXPERIENCE_WEIGHTS),
    })

    # Hidden motivation (not saved): personal trait plus a channel shift
    motivation = rng.normal(0, 1, n) + np.array([CHANNEL_MOTIVATION[c] for c in channel])
    return learners, motivation


def generate_enrollments(rng: np.random.Generator, learners: pd.DataFrame,
                         courses: pd.DataFrame, motivation: np.ndarray) -> pd.DataFrame:
    """Create the enrollments table with engagement metrics and the outcome."""
    n_l, n_c, n = len(learners), len(courses), N_ENROLLMENTS

    # Unique (learner, course) pairs; some learners and courses are more active/popular
    learner_activity = rng.gamma(2.0, 1.0, n_l)
    course_popularity = rng.gamma(3.0, 1.0, n_c)
    pair_p = np.outer(learner_activity, course_popularity).ravel()
    pair_p /= pair_p.sum()
    pair_idx = rng.choice(n_l * n_c, size=n, replace=False, p=pair_p)
    l_idx, c_idx = pair_idx // n_c, pair_idx % n_c

    # Enrollment date: shortly after signup (exponential delay, capped)
    signup = pd.to_datetime(learners["signup_date"]).iloc[l_idx].reset_index(drop=True)
    delay = np.minimum(rng.exponential(14, n).astype(int), MAX_ENROLL_DELAY_DAYS)
    enroll_date = signup + pd.to_timedelta(delay, unit="D")

    course = courses.iloc[c_idx].reset_index(drop=True)
    learner = learners.iloc[l_idx].reset_index(drop=True)
    m = motivation[l_idx]
    n_lessons = course["n_lessons"].to_numpy()
    duration = course["duration_weeks"].to_numpy()

    # Completion probability (logistic model on hidden and observed factors)
    logit = (-0.35 + 0.9 * m
             + course["difficulty"].map(DIFFICULTY_EFFECT).to_numpy()
             - 0.4 * course["is_free"].to_numpy()
             + learner["device"].map(DEVICE_EFFECT).to_numpy()
             + learner["prior_experience"].map(EXPERIENCE_EFFECT).to_numpy()
             + rng.normal(0, 0.5, n))
    completed = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    # Early activity is related to motivation and to the final outcome
    first_week_sessions = rng.poisson(np.exp(0.6 + 0.35 * m + 0.5 * completed))

    # Lessons completed: all lessons if completed, otherwise a partial amount
    partial = rng.beta(2.0, 2.5, n) * 95 / 100
    lessons = np.where(completed == 1, n_lessons, np.round(partial * n_lessons)).astype(int)
    lessons = np.where(completed == 1, lessons, np.minimum(lessons, n_lessons - 1))
    lessons = np.where((completed == 0) & (first_week_sessions == 0), 0, lessons)
    progress_pct = (lessons / n_lessons * 100).round(1)

    # Other engagement metrics
    n_sessions = first_week_sessions + rng.poisson(0.5 * lessons)
    watch_minutes = (lessons * rng.lognormal(np.log(11), 0.35, n)
                     + (first_week_sessions > 0) * rng.gamma(2.0, 5.0, n)).round(0)
    forum_posts = rng.poisson(np.exp(-1.0 + 0.5 * m + 0.4 * completed))

    quiz = np.clip(rng.normal(68 + 7 * m + 4 * completed, 10), 0, 100).round(1)
    no_quiz = (lessons < 3) | (rng.random(n) < 0.10)
    quiz = np.where(no_quiz, np.nan, quiz)

    # Week of last activity for learners who did not finish
    drop_week = np.clip(np.ceil(progress_pct / 100 * duration), 1, duration).astype(int)
    dropout_week = pd.array(np.where(completed == 1, -1, drop_week), dtype="Int64")
    dropout_week = dropout_week.copy()
    dropout_week[completed == 1] = pd.NA

    enrollments = pd.DataFrame({
        "learner_id": learner["learner_id"].to_numpy(),
        "course_id": course["course_id"].to_numpy(),
        "enroll_date": enroll_date.dt.strftime("%Y-%m-%d"),
        "first_week_sessions": first_week_sessions,
        "n_sessions": n_sessions,
        "lessons_completed": lessons,
        "progress_pct": progress_pct,
        "watch_minutes": watch_minutes.astype(int),
        "quiz_avg_score": quiz,
        "forum_posts": forum_posts,
        "completed": completed,
        "dropout_week": dropout_week,
    })

    # Order by enrollment date and assign sequential ids
    enrollments = enrollments.sort_values(["enroll_date", "learner_id", "course_id"])
    enrollments = enrollments.reset_index(drop=True)
    enrollments.insert(0, "enrollment_id", np.arange(1, n + 1))
    return enrollments


def main() -> None:
    """Generate all tables and save them to data/."""
    rng = np.random.default_rng(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)

    courses = generate_courses(rng)
    learners, motivation = generate_learners(rng)
    enrollments = generate_enrollments(rng, learners, courses, motivation)

    tables = {"learners": learners, "courses": courses, "enrollments": enrollments}
    for name, df in tables.items():
        path = DATA_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {name}: {len(df)} rows, {df.shape[1]} columns -> {path.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
