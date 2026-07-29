import numpy as np
from collections import Counter

def knn_classify(X_train, y_train, X_test, k=3):
    """
    Returns: A list of predicted integer labels for each test point
    """
    X_train =np.array(X_train)
    y_train =np.array(y_train)
    X_test =np.array(X_test)
    predictions =[]
    for x in X_test:
        md =np.sum((X_train-x)**2, axis=1)
        dt = np.sqrt(md)
        near_neighbor = np.argsort(dt)[:k] 
        # argsortจะคืนindexทั้งหมดnตัว^ เรียงจากใกล้ไปไกล
        #ส่วน k คือการ slice เอาแค่ k ตัวแรก^
        k_labels = y_train[near_neighbor]
        ct = Counter(k_labels)
        """
        โจทย์บอกว่า ถ้าเสมอ ให้เลือก label ที่เลขน้อยที่สุด
        ดังนั้นต้องหาวิธีบังคับ logic นี้เอง
        แนวคิดวิธีแก้
        1. โหวตเยอะสุด (primary criteria)
            -> เรียงตามจำนวนโหวตจากมากไปน้อยเป็นหลัก
        2. ถ้าโหวตเท่ากัน ให้เลือกเลขน้อยสุด (tie-breaker)
            ->ถ้าโหวตเท่ากัน อยากให้labelเลขน้อยกว่าถูกเลือก (ซึ่งหมายถึงต้อง "แปลง" label ให้เป็นตัวเลขที่เมื่อเทียบกันแล้ว เลขน้อยกว่าดูเหมือน "ชนะ" ในการเปรียบเทียบรอง)

        วิธีคิดแบบหนึ่งคือ ใช้ max() พร้อม key func. พิเศษ
        """
        best_label = max(ct.keys(), key = lambda label:(ct[label],-label))
        predictions.append(best_label)
    return predictions