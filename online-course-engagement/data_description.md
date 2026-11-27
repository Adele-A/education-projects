# Data Description

All data is **synthetic**. It is produced by `src/data_generator.py` (numpy, seed 42) and saved to `data/`. It does not represent real people or courses.

The generator uses a hidden per-learner "motivation" score that is **not** saved. It influences completion and engagement together with learner and course attributes.

Table relationships: `enrollments.learner_id` references `learners.learner_id`, and `enrollments.course_id` references `courses.course_id`. Each (learner, course) pair appears at most once.

## learners.csv (2000 rows)

| Column | Type | Meaning | Value range |
|--------|------|---------|-------------|
| learner_id | int | Unique learner identifier (primary key) | 1 to 2000 |
| signup_date | date (YYYY-MM-DD) | Date the learner registered | 2023-01-01 to 2024-12-31 |
| country | string | Learner country | United States, India, United Kingdom, Germany, Brazil, Canada, Nigeria, Poland |
| age | int | Age in years | 18 to 65 |
| device | string | Main device used | Mobile, Desktop, Tablet |
| acquisition_channel | string | How the learner found the platform | Organic Search, Paid Ads, Referral, Social Media, Email |
| prior_experience | string | Self-reported experience in the subject area | No experience, Some experience, Experienced |

## courses.csv (20 rows)

| Column | Type | Meaning | Value range |
|--------|------|---------|-------------|
| course_id | int | Unique course identifier (primary key) | 1 to 20 |
| title | string | Course title | 20 distinct titles |
| category | string | Subject area | Programming, Data Science, Business, Design, Marketing |
| difficulty | string | Difficulty level | Beginner, Intermediate, Advanced |
| duration_weeks | int | Planned course length in weeks | 4 to 12 (Beginner 4-6, Intermediate 5-8, Advanced 8-12) |
| n_lessons | int | Number of lessons in the course | 10 to 60 (Beginner 10-25, Intermediate 20-40, Advanced 30-60) |
| is_free | int (0/1) | 1 if the course is free | 0 or 1 |
| price_usd | int | Course price in US dollars | 0 for free courses; otherwise one of 29, 49, 79, 99, 149 |

## enrollments.csv (4500 rows)

| Column | Type | Meaning | Value range |
|--------|------|---------|-------------|
| enrollment_id | int | Unique enrollment identifier (primary key), assigned in order of enrollment date | 1 to 4500 |
| learner_id | int | Learner who enrolled (foreign key to learners) | 1 to 2000 |
| course_id | int | Course enrolled in (foreign key to courses) | 1 to 20 |
| enroll_date | date (YYYY-MM-DD) | Date of enrollment; 0 to 90 days after the learner's signup date | 2023-01-01 to 2025-03-31 |
| first_week_sessions | int | Number of learning sessions in the first 7 days after enrollment | 0 or more (typically 0 to about 10) |
| n_sessions | int | Total learning sessions during the course (at least first_week_sessions) | 0 or more |
| lessons_completed | int | Number of lessons finished | 0 to n_lessons of the course |
| progress_pct | float | lessons_completed / n_lessons * 100, rounded to 1 decimal | 0.0 to 100.0 |
| watch_minutes | int | Total video watch time in minutes | 0 or more |
| quiz_avg_score | float | Average quiz score; missing if the learner completed fewer than 3 lessons or took no quizzes (about 10% of other rows are also missing at random) | 0.0 to 100.0, or empty |
| forum_posts | int | Number of forum posts written | 0 or more (mostly small values) |
| completed | int (0/1) | Target-related outcome: 1 if the learner finished the course (progress_pct = 100), 0 otherwise (dropout) | 0 or 1 |
| dropout_week | Int (nullable) | Week of the course in which a non-completing learner was last active; empty for completers | 1 to duration_weeks of the course, or empty |

Notes:
- Dropout is defined as `completed = 0`.
- `first_week_sessions` is the main early-warning signal available one week after enrollment. Columns such as `lessons_completed`, `progress_pct`, `n_sessions`, `watch_minutes`, `quiz_avg_score` and `dropout_week` describe the whole course and would cause data leakage if used to predict dropout at the end of week 1.
