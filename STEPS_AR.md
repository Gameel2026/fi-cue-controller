# خطوات تشغيل ورقة الـ Freeze Index (من السبت للأربعاء)

**قاعدة مهمة:** لحد الخطوة 9 (التجميد ورفعه)، ما تفتحيش ولا تحللي أي dataset خارجية (FoG-STAR أو Stanford أو Multimodal). الشغل كله على tDCS-FoG وDeFOG وDaphnet بس.

الكود كله في الفولدر `C:\Users\engom\Downloads\fi_paper`. كل الأوامر بتتكتب في **Anaconda Prompt**.

---

## اليوم الأول (السبت)

### الخطوة 1: ارفعي الخطة على GitHub قبل أي تشغيل (10 دقائق)
1. ادخلي على GitHub واعملي repository جديد اسمه `fi-cue-controller`، يكون **Public**، ومن غير README.
2. اضغطي **uploading an existing file**.
3. ارفعي كل ملفات الفولدر `fi_paper` (ملف `PLAN.md` وكل ملفات `.py`، والملفين `requirements.txt` و`STEPS_AR.md`).
4. اكتبي في الـ commit message:
   `Development plan and code, committed before the development analyses`
5. اضغطي **Commit changes**، وانسخي لينك الـ commit واحتفظي بيه.

### الخطوة 2: افحصي البيئة (دقيقة)
```
cd C:\Users\engom\Downloads\fi_paper
python s0_check.py
```
لازم يطلع في الآخر `ALL GOOD`.
- لو ظهر `MISSING` لأي مكتبة، اكتبي: `pip install numpy scipy pandas scikit-learn matplotlib`
- سطر الـ torch مش مهم في اليوم الأول.

### الخطوة 3: تجهيز البيانات (حوالي 15–30 دقيقة)
```
python s1_prepare.py
```
في الآخر هيطبع جدول فيه 3 صفوف: daphnet وdefog وtdcsfog.

### الخطوة 4: حساب الـ features (حوالي 10–20 دقيقة)
```
python s2_features.py
```
لازم تطلع 3 سطور فيها كلمة `saved`.

### الخطوة 5: تجربة الـ controllers من D0 لـ D5 (حوالي 30–90 دقيقة)
```
python s3_ladder.py
```
- هيطبع `fold 1 / 5 ... fold 5 / 5` لكل dataset، وفي الآخر جدول النتايج.
- سيبي اللابتوب شغال وموصّل بالشاحن، وما تقفليش الشاشة السوداء لحد ما تخلص.

---

## اليوم الثاني (الأحد)

### الخطوة 6: الموديل المتعلّم على الـ GPU (حوالي 30–90 دقيقة)
1. شغّلي `python s0_check.py` تاني، وبصي على سطر الـ torch:
   - لو مكتوب `CUDA available: True`، كملي على النقطة 2.
   - لو مكتوب `False` أو `torch not available`، اكتبي:
     ```
     pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
     ```
     وبعدين شغّلي `python s0_check.py` تاني.
   - لو التسطيب فشل، اكتبي `pip install torch==1.13.1`. ده هيشتغل على الـ CPU، بس أبطأ (ممكن 3–5 ساعات).
2. شغّلي:
   ```
   python s4_cnn.py
   ```
   هيطبع `device: cuda` (أو `cpu`)، وبعدين الـ loss لكل epoch، وفي الآخر جدول.
3. **لو الخطوة دي فشلت خالص، ما تقلقيش.** هي مرجع اختياري، والورقة ماشية من غيرها. كملي على الخطوة 7.

### الخطوة 7: تحليل الفشل ومقارنته بالـ chance (حوالي 30–60 دقيقة)
```
python s5_report.py
```
هيطبع جداول:
- **coverage حسب kinetic / akinetic:** هنا بالظبط بيظهر فين الـ FI بيفشل.
- **coverage حسب نوع الـ FoG.**
- **الـ false alarms حسب المهمة.**

لو الخطوة دي أخدت وقت أطول من ساعتين، اقفليها بـ `Ctrl + C` وشغّليها بعدد shifts أقل:
```
python s5_report.py --shifts=300
```

---

**لو خلصتي بدري في أي يوم، كملي على خطوات اليوم اللي بعده على طول.** الترتيب هو المهم، مش اليوم.

## اليوم الثالث (الاثنين): تجميد الإعدادات

### الخطوة 8: التجميد (حوالي 20–60 دقيقة)
لو الخطوة 6 (الـ CNN) اشتغلت معاكي، شغّلي:
```
python s6_freeze.py --cnn
```
ولو ما اشتغلتش، شغّلي:
```
python s6_freeze.py
```
- الأمر ده بيختار الـ thresholds النهائية من كل بيانات التطوير بنفس القاعدة المكتوبة في الخطة، ويحفظها في `results_fi\frozen\frozen_fi.json`.
- في الآخر هيطبع سطر فيه `sha256 of frozen_fi.json`. انسخيه واحتفظي بيه.

### الخطوة 9: ارفعي التجميد على GitHub (10 دقائق)
1. في نفس الـ repo `fi-cue-controller` اعملي **Add file** ثم **Upload files**.
2. ارفعي كل محتويات فولدر `results_fi\frozen` (ملف `frozen_fi.json` وملفات `lr_lab.json` و`lr_home.json`، و`cnn_lab.pt` و`cnn_home.pt` لو موجودين).
3. ارفعي كمان ملفي `s6_freeze.py` و`s7_external_review.py`.
4. اكتبي في الـ commit message:
   `Frozen development configuration, committed before any external data are analysed`
5. انسخي لينك الـ commit ده.

**مهم:** التجميد ده بيقفل الإعدادات قبل ما نلمس أي dataset خارجية. لو قررنا يوم التلات نعدل حاجة بعد مراجعة النتايج، هنعمل amendment بتاريخ وسبب، وده مسموح في الخطة.

---

## اليوم الرابع (الثلاثاء): تجهيز البيانات الخارجية من غير أي اختبار

### الخطوة 10: نزّلي dataset Stanford (10 دقائق)
1. افتحي https://github.com/stanfordnmbl/imu-fog-detection
2. اضغطي الزرار الأخضر **Code** ثم **Download ZIP** (حوالي 100–200 MB).
3. اعملي فولدر جديد: `D:\New research fog work\Stanford dataset`
4. فكّي الـ ZIP جوه الفولدر ده. لازم يبقى عندك المسار ده بالظبط وجواه 60 ملف `pt...xlsx`:
   `D:\New research fog work\Stanford dataset\imu-fog-detection-main\data\raw\imus6_subjects7`
5. لو لقيتي الملفات حجمها صغير جداً (حوالي 1 KB) أو مش بتتفتح، قوليلي يوم التلات. ده معناه إنهم محتاجين طريقة تنزيل تانية.

### الخطوة 11: مراجعة شكل البيانات الخارجية (حوالي 10–20 دقيقة)
لو ظهر في الخطوة 2 إن `openpyxl` ناقصة، اكتبي `pip install openpyxl` الأول. وبعدين:
```
python s7_external_review.py
```
- الأمر ده بيقرا FoG-STAR وMultimodal وStanford، ويحوّلهم لنفس الشكل.
- وبيطبع معلومات عن شكل البيانات بس: عدد التسجيلات، وعدد نوبات الـ FoG، واتجاه المحاور.
- **مش بيشغّل أي controller ومش بيحسب أي أداء.** الاختبار الخارجي الحقيقي هنكتبه ونشغّله مع بعض بعد ما أراجع التجميد ونتايج المراجعة دي.
- النتيجة بتتحفظ في `results_fi\external_review.txt`.

---

## اليوم الخامس (الأربعاء): التسليم

### الخطوة 12: جهّزي كل حاجة عشان تبعتيها
1. افتحي فولدر `C:\Users\engom\Downloads\fi_paper\results_fi`.
2. اعملي له Zip: كليك يمين، وبعدين **Send to**، وبعدين **Compressed (zipped) folder**.
3. لما نتكلم تاني، ابعتيلي:
   - ملف الـ zip ده.
   - لينك الـ commit بتاع الخطوة 1، ولينك الـ commit بتاع الخطوة 9.
   - Screenshot أو نسخ لأي رسالة error ظهرت.

**ما ترفعيش نتايج التطوير على GitHub دلوقتي** (غير ملفات التجميد في الخطوة 9). هنراجعها مع بعض الأول.
**وما تشغّليش أي اختبار على البيانات الخارجية** غير الخطوة 11.

---

## لو حصلت مشكلة
- **`FileNotFoundError`:** افتحي `fi_config.py` بالـ Notepad، واتأكدي إن المسارات مظبوطة:
  - `KAGGLE_ROOT` لازم يكون الفولدر اللي جواه `train` و`events.csv`.
  - `DAPHNET_ROOT` لازم يكون الفولدر اللي جواه `S01R01.txt`.
- **`MemoryError`:** اقفلي البرامج التانية وشغّلي الخطوة تاني.
- **أي error تاني:** صوّريه وكملي الخطوات اللي بعده لو ممكن (الخطوة 6 مستقلة عن 7). وابعتيلي الصورة يوم التلات.

## ترتيب الملفات (للعلم)
| الملف | وظيفته |
|---|---|
| `fi_config.py` | المسارات والإعدادات |
| `s0_check.py` | فحص البيئة والبيانات |
| `s1_prepare.py` | تحويل البيانات لـ 64 Hz وبوحدة g، مع labels نوع الـ FoG والمهام |
| `s2_features.py` | حساب الـ Freeze Index والـ features كل 0.25 ثانية (causal) |
| `s3_ladder.py` | الـ controllers من D0 لـ D5 بـ subject-grouped 5-fold CV |
| `s4_cnn.py` | D6: موديل CNN صغير causal كمرجع متعلّم |
| `s5_report.py` | تحليل الفشل حسب النوع والمهمة، ومقارنة بالـ chance |
| `s6_freeze.py` | تجميد الإعدادات النهائية من كل بيانات التطوير |
| `s7_external_review.py` | تجهيز البيانات الخارجية ومراجعة شكلها بس، من غير اختبار |
