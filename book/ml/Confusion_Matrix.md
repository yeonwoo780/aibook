# 혼동 행렬(Confusion Matrix)

## 목차

- [Confusion Matrix란?](#confusion-matrix란)
- [Accuracy](#Accuracy)
- [Precision](#Precision)
- [Recall](#Recall)
- [QA](#QA)

---

### Confusion Matrix란

참과 거짓 중 하나를 예측하는 문제였다고 가정해보자. 아래의 혼동 행렬에서 각 열은 예측값을 나타내며, 각 행은 실제값을 나타낸다.

|       | 예측 참 | 예측 거짓 |
| ----- | ---- | ----- |
| 실제 참  | TP   | FN    |
| 실제 거짓 | FP   | TN    |

다음과 같은 네가지 케이스에 대해 각각 TP, FP, FN, TN을 정의한다.
- TP(True Positive) : 실제 True인 정답을 True라고 예측(정답)
- FP(False Positive) : 실제 False인 정답을 True라고 예측(오답)
- FN(False Negative) : 실제 True인 정답을 False라고 예측(오답)
- TN(True Negative) : 실제 False인 정답을 False라고 예측(정답)

---

### Accuracy
Accuracy(정확도)란 판별한 전체 샘플 중 TP와 TN의 비율이다. 분류 모델을 평가하기에 가장 단순한 지표이지만, 불균형한 클래스를 가진 데이터셋을 평가하기 어렵다는 단점이 있다.
예를 들어, Positive와 Negative의 비율이 2:8로 불균형한 클래스를 가지는 데이터셋에서는 모든 예측을 Negative로 해버리는 엉터리 분류기의 정확도도 80%로 측정된다.

Accuracy = (TP + TN)/(TP + FP + FN + TN)

---

### Precision
Precision(정밀도)란 분류 모델이 Positive로 판정한 것 중, 실제로 Positive인 샘플의 비율이다. Precision은 Positive로 검출된 결과가 얼마나 정확한지를 나타낸다.

Precision = TP/(TP + FP)

---

### Recall
Recall(재현율)이란 실제 Positive 샘플 중 분류 모델이 Positive로 판정한 비율이다. Recall은 분류 모델이 실제 Positive 클래스를 얼마나 빠지지 않고 잘 잡아내는지를 나타낸다.

Recall = TP/(TP + FN)

---

### QA

#### Q. 비지도 학습에서 사용되는 데이터의 특징은?

1. 레이블이 명시되어 있다.
2. 레이블이 명시되어 있지 않다.
3. 레이블이 선택적으로 사용된다.
4. 레이블이 항상 동일하다.
   
   <details><summary>정답</summary>

2

</details>
