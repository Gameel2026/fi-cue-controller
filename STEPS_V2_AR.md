# خطوات النسخة v2 (بعد ما خلصتي خطوات v1)

كل الأوامر في **Anaconda Prompt** من جوه الفولدر:
```
cd C:\Users\engom\Downloads\fi_paper
```

**القاعدة الذهبية:** ما تشغليش الخطوة 4 ولا الخطوة 5 قبل ما ترفعي الخطوة 3 على GitHub.

---

### الخطوة 1: ارفعي الـ Amendment والكود الجديد على GitHub الأول (5 دقائق)
1. في الـ repo `fi-cue-controller` اختاري **Add file** ثم **Upload files**.
2. ارفعي الملفات دي:
   - `AMENDMENT_1.md`
   - `STEPS_V2_AR.md`
   - `fi2_features.py`
   - `s11_v2_develop.py`
   - `s12_kaggle_test_prep.py`
   - `s13_v2_test.py`
3. اكتبي في الـ commit message:
   `Amendment 1: v2 controller plan and code, committed before v2 development`
4. احتفظي بلينك الـ commit.

### الخطوة 2: طوّري v2 وجمّديه (حوالي 30–90 دقيقة)
```
python s11_v2_develop.py
```
- الأمر ده بيحسب features جديدة، ويدرّب الموديل بـ 5-fold CV على بيانات التطوير بس، وفي الآخر بيجمّده تلقائياً.
- بيطبع جدول النتايج، وبعده سطر فيه `sha256 of frozen_v2.json`. **انسخي الرقم ده.**
- لو ظهر error فيه كلمة `HistGradientBoostingClassifier`، اكتبي `pip install -U scikit-learn` وعيدي الخطوة.

### الخطوة 3: ارفعي التجميد على GitHub قبل أي اختبار (5 دقائق)
1. ارفعي ملفين من فولدر `results_fi\v2`:
   - `frozen_v2.json`
   - `v2_model.pkl`
2. ارفعي كمان `v2_cv_summary.csv` من نفس الفولدر.
3. اكتبي في الـ commit message:
   `Frozen v2 configuration, committed before any v2 test`
4. احتفظي بلينك الـ commit.

### الخطوة 4: جهّزي بيانات الاختبار الجديدة من Kaggle (حوالي 15–30 دقيقة)
```
python s12_kaggle_test_prep.py
```
- الأمر ده بيقرا ملفات الـ test (public وprivate) والـ labels بتاعتها من `official solution.csv` (ملف كبير، فالخطوة هتاخد وقت).
- بيشيل لوحده الملفات اللي مرضاها ظهروا في بيانات التدريب.
- في الآخر بيطبع جدول فيه عدد الملفات والمرضى في `kaggle_test_tdcsfog` و`kaggle_test_defog`.

### الخطوة 5: الاختبار (مرة واحدة بس) (حوالي 30–60 دقيقة)
```
python s13_v2_test.py
```
- بيختبر v2 والنسخة v1 المجمدة على بيانات Kaggle الجديدة. وده الاختبار الأساسي لـ v2.
- وكمان بيطبّق v2 على FoG-STAR وMultimodal وStanford، ودي هتتكتب في الورقة كتحليل ثانوي.
- أول سطرين بيطبعوا رقمين sha256. **التاني لازم يكون نفس الرقم اللي نسختيه في الخطوة 2.**
- **شغّليه مرة واحدة بس.** لو ظهر error، صوّريه وما تعدّليش وتعيدي.

### الخطوة 6: التسليم
1. اعملي Zip لفولدر `C:\Users\engom\Downloads\fi_paper\results_fi\v2`.
2. ابعتيلي:
   - ملف الـ zip.
   - لينكات الـ commits بتاعة الخطوة 1 والخطوة 3.
   - صورة لأي error ظهر.

---

## لو حصلت مشكلة
- **`MemoryError` في الخطوة 2:** افتحي `s11_v2_develop.py` بالـ Notepad، وغيّري `MAX_TRAIN = 600000` لـ `MAX_TRAIN = 300000`، واحفظي، وعيدي الخطوة. ولو عملتي التغيير ده، ارفعي الملف المعدّل على GitHub قبل الخطوة 3.
- **الخطوة 4 طبعت `no_labels` كتير:** كملي عادي، وقوليلي الرقم.
