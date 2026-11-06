# 1. IMPORT LIBRARIES

# تقليل استهلاك الذاكرة ومنع مشكلة OpenBLAS
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# استيراد المكتبات المستخدمة في المشروع
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# 2. LOAD DATASET
# قراءة ملف البيانات
df = pd.read_csv("train.csv")

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== DATASET INFO ==========")
df.info()

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== SURVIVAL DISTRIBUTION ==========")
print(df["Survived"].value_counts())

print("\n========== PASSENGERS BY GENDER ==========")
print(df["Sex"].value_counts())

print("\n========== SURVIVAL RATE BY GENDER ==========")
print(df.groupby("Sex")["Survived"].mean())


# 3. SQLITE DATABASE
# إنشاء قاعدة البيانات وحفظ بيانات الركاب
conn = sqlite3.connect("titanic.db")

df.to_sql("passengers", conn, if_exists="replace", index=False)

conn.close()

print("\nDataset saved successfully in SQLite!")

print("Database exists:", os.path.exists("titanic.db"))

print("Database size:", os.path.getsize("titanic.db"), "bytes")


# التأكد من الجداول الموجودة في قاعدة البيانات
conn = sqlite3.connect("titanic.db")
cursor = conn.cursor()

cursor.execute("""
SELECT name
FROM sqlite_master
WHERE type='table';
""")

print("\nTables in Database:", cursor.fetchall())

# حساب عدد الركاب
cursor.execute("SELECT COUNT(*) FROM passengers")

count = cursor.fetchone()[0]

print("Total Passengers:", count)

conn.close()


# قراءة أول خمس سجلات من قاعدة البيانات
conn = sqlite3.connect("titanic.db")

query = """
SELECT *
FROM passengers
LIMIT 5;
"""

result = pd.read_sql_query(query, conn)

print("\n========== DATABASE PREVIEW ==========")
print(result)

conn.close()


# تحليل بيانات النجاة باستخدام SQL
conn = sqlite3.connect("titanic.db")

query = """
SELECT
    COUNT(*) AS total_passengers,
    SUM(Survived) AS survived_passengers,
    ROUND(AVG(Survived) * 100, 2) AS survival_rate
FROM passengers;
"""

result = pd.read_sql_query(query, conn)

print("\n========== SQL SURVIVAL ANALYSIS ==========")
print(result)

conn.close()

# 4. DATA CLEANING

# إنشاء نسخة من البيانات الأصلية
df_clean = df.copy()

# تعويض القيم المفقودة في العمر باستخدام الوسيط
df_clean["Age"] = df_clean["Age"].fillna(
    df_clean["Age"].median()
)

# تعويض القيم المفقودة في ميناء المغادرة باستخدام القيمة الأكثر تكراراً
df_clean["Embarked"] = df_clean["Embarked"].fillna(
    df_clean["Embarked"].mode()[0]
)

# حذف عمود Cabin بسبب كثرة القيم المفقودة
df_clean = df_clean.drop(columns=["Cabin"])

print("\n========== MISSING VALUES AFTER CLEANING ==========")
print(df_clean.isnull().sum())

print("\nCleaned Dataset Shape:")
print(df_clean.shape)


# 5. DATA PREPROCESSING

# إنشاء نسخة جديدة لتجهيز البيانات للنماذج
df_encoded = df_clean.copy()

# تحويل الجنس إلى قيم رقمية
# ذكر = 0 ، أنثى = 1
df_encoded["Sex"] = df_encoded["Sex"].map({
    "male": 0,
    "female": 1
})

# تحويل ميناء المغادرة إلى أعمدة رقمية
df_encoded = pd.get_dummies(
    df_encoded,
    columns=["Embarked"],
    dtype=int
)

print("\n========== ENCODED DATASET ==========")
print(df_encoded.head())

print("Encoded Shape:", df_encoded.shape)

# 6. FEATURE SELECTION

# تحديد الخصائص المستخدمة في التدريب
# حذف الأعمدة غير المناسبة للتنبؤ
X = df_encoded.drop(
    columns=["Survived", "PassengerId", "Name", "Ticket"]
)

# تحديد العمود المطلوب التنبؤ به
y = df_encoded["Survived"]

print("\nFeatures Shape:", X.shape)
print("Label Shape:", y.shape)

print("\nSelected Features:")
print(X.columns.tolist())

# 7. TRAIN TEST SPLIT

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n========== TRAIN TEST SPLIT ==========")

print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)

print("Training Labels:", y_train.shape)
print("Testing Labels:", y_test.shape)

# 8. LOGISTIC REGRESSION


# إنشاء وتدريب نموذج Logistic Regression
log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train, y_train)

# إجراء التنبؤات
log_predictions = log_model.predict(X_test)

print("\nLogistic Regression trained successfully!")


# 9. ORIGINAL DECISION TREE

# إنشاء شجرة القرار الأصلية بعمق أقصى يساوي 5
tree_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

tree_model.fit(X_train, y_train)

tree_predictions = tree_model.predict(X_test)

print("Decision Tree trained successfully!")


# 10. MODEL EVALUATION


# تقييم نموذج Logistic Regression
print("\n========== LOGISTIC REGRESSION ==========")

print("Accuracy:",
    accuracy_score(y_test, log_predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, log_predictions))

print("\nClassification Report:")
print(classification_report(y_test, log_predictions))


# تقييم شجرة القرار الأصلية
print("\n========== ORIGINAL DECISION TREE ==========")

print("Accuracy:",
    accuracy_score(y_test, tree_predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, tree_predictions))

print("\nClassification Report:")
print(classification_report(y_test, tree_predictions))

# 11. INITIAL OVERFITTING ANALYSIS

# حساب دقة النماذج على بيانات التدريب
log_train_predictions = log_model.predict(X_train)

tree_train_predictions = tree_model.predict(X_train)

log_train_accuracy = accuracy_score(
    y_train,
    log_train_predictions
)

tree_train_accuracy = accuracy_score(
    y_train,
    tree_train_predictions
)

print("\n========== INITIAL OVERFITTING ANALYSIS ==========")

print("Logistic Regression Training Accuracy:",
    log_train_accuracy)

print("Decision Tree Training Accuracy:",
    tree_train_accuracy)


# 12. MAX DEPTH EXPERIMENTS

# تجربة أعماق مختلفة لشجرة القرار
depths = [2, 3, 4, 5, 6, 8, 10]

print("\n========== MAX DEPTH EXPERIMENTS ==========")

for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)

    test_predictions = model.predict(X_test)

    train_accuracy = accuracy_score(
        y_train,
        train_predictions
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    print(f"\nMax Depth: {depth}")

    print(f"Training Accuracy: {train_accuracy:.4f}")

    print(f"Testing Accuracy: {test_accuracy:.4f}")

# 13. GRID SEARCH CV - OPTIMIZATION


# تحديد النموذج والإعدادات المطلوب تجربتها
dt = DecisionTreeClassifier(random_state=42)

param_grid = {
    "max_depth": [3, 4, 5, 6, 8, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# استخدام Grid Search لاختيار أفضل الإعدادات
grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=1
)

print("\n========== GRID SEARCH STARTED ==========")

grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross Validation Accuracy:")
print(grid_search.best_score_)

# 14. OPTIMIZED DECISION TREE


# استخراج أفضل نموذج بعد عملية التحسين
best_tree = grid_search.best_estimator_

best_predictions = best_tree.predict(X_test)


# 15. OPTIMIZED MODEL EVALUATION


print("\n========== OPTIMIZED DECISION TREE ==========")

print("Best Decision Tree Test Accuracy:")

print(accuracy_score(y_test, best_predictions))

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, best_predictions))

print("\nClassification Report:")

print(classification_report(y_test, best_predictions))


# 16. CONFUSION MATRIX VISUALIZATION


# رسم مصفوفة الارتباك للنموذج المحسن
ConfusionMatrixDisplay.from_predictions(
    y_test,
    best_predictions,
    display_labels=["Not Survived", "Survived"],
    cmap="Blues"
)

plt.title("Optimized Decision Tree - Confusion Matrix")

plt.tight_layout()

plt.show(block=False)
plt.pause(2)
plt.close()

# 17. DECISION TREE VISUALIZATION

# رسم شجرة القرار المحسنة
plt.figure(figsize=(24, 12))

plot_tree(
    best_tree,
    feature_names=X_train.columns,
    class_names=["Not Survived", "Survived"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Optimized Decision Tree")

plt.tight_layout()

plt.show()


# 18. DECISION TREE STRUCTURE

# عرض بنية شجرة القرار والخاصية المستخدمة في أول تقسيم
print("\n========== DECISION TREE STRUCTURE ==========")

print("\nSex Encoding:")

print(df["Sex"].value_counts())

print("\nFeature Order:")

print(X_train.columns.tolist())

root_feature_index = best_tree.tree_.feature[0]

if root_feature_index >= 0:

    root_feature_name = X_train.columns[root_feature_index]

    print("\nRoot Feature:", root_feature_name)

    print("Root Threshold:", best_tree.tree_.threshold[0])


# 19. OPTIMIZED TREE OVERFITTING ANALYSIS

# حساب دقة النموذج المحسن على بيانات التدريب والاختبار
best_train_predictions = best_tree.predict(X_train)

best_test_predictions = best_tree.predict(X_test)

best_train_accuracy = accuracy_score(
    y_train,
    best_train_predictions
)

best_test_accuracy = accuracy_score(
    y_test,
    best_test_predictions
)

accuracy_gap = best_train_accuracy - best_test_accuracy

print("\n========== OPTIMIZED TREE OVERFITTING ANALYSIS ==========")

print("Training Accuracy:", best_train_accuracy)

print("Testing Accuracy:", best_test_accuracy)

print("Accuracy Gap:", accuracy_gap)


# 20. MODEL COMPARISON

# مقارنة أداء النماذج الثلاثة
models_comparison = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Original Decision Tree",
        "Optimized Decision Tree"
    ],

    "Training Accuracy": [
        log_train_accuracy,
        tree_train_accuracy,
        best_train_accuracy
    ],

    "Testing Accuracy": [
        accuracy_score(y_test, log_predictions),
        accuracy_score(y_test, tree_predictions),
        best_test_accuracy
    ]
})

print("\n========== MODEL COMPARISON ==========")

print(models_comparison.to_string(index=False))


# تحديد النموذج صاحب أعلى دقة اختبار
best_model_index = models_comparison["Testing Accuracy"].idxmax()

print("\nBest Model:")

print(models_comparison.loc[best_model_index, "Model"])

print("\nBest Testing Accuracy:")

print(models_comparison.loc[best_model_index, "Testing Accuracy"])

# 21. FINAL PROJECT SUMMARY

print("\n========================================")

print("TITANIC SURVIVAL PREDICTION")

print("PHANTOMS CAPSTONE PROJECT")

print("========================================")

print("\nDataset Size:", df.shape)

print("Number of Training Samples:", len(X_train))

print("Number of Testing Samples:", len(X_test))

print("\nBest Model:",
    models_comparison.loc[best_model_index, "Model"])

print("Best Accuracy:",
      round(models_comparison.loc[best_model_index, "Testing Accuracy"] * 100, 2),
    "%")

print("\n========================================")

print("TITANIC CAPSTONE EXECUTION COMPLETED!")

print("========================================")

print("\n========== MODEL COMPARISON ==========")

print("Logistic Regression:")
print("Training:", log_train_accuracy)
print("Testing:", accuracy_score(y_test, log_predictions))

print("\nOriginal Decision Tree:")
print("Training:", tree_train_accuracy)
print("Testing:", accuracy_score(y_test, tree_predictions))

print("\nOptimized Decision Tree:")
print("Training:", best_train_accuracy)
print("Testing:", best_test_accuracy)

print("\n========== PROJECT COMPLETED ==========")