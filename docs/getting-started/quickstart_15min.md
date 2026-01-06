# 15-Minute Quickstart

**Purpose:** Get FoodSpec working on your computer and run your first analysis in 15 minutes.

**Audience:** Complete beginners with zero spectroscopy experience.

**Time:** 15 minutes exactly.

**Prerequisites:** Python 3.10+; basic terminal/command-line knowledge; no chemistry background needed.

**What you'll accomplish:**
1. Install FoodSpec
2. Download sample oil spectroscopy data
3. Run oil authentication analysis
4. View results (confusion matrix, metrics)

---

## What You'll Learn

By the end of this guide, you will:
1. Install FoodSpec on your computer
2. Download a sample food dataset (cooking oils)
3. Run oil authentication analysis
4. View results (confusion matrix, ROC curve, metrics)

**No equations. No chemistry jargon. Just working code.**

---

## Step 1: Install FoodSpec

⏱️ 2 minutes

Open your terminal (Mac/Linux) or Command Prompt (Windows) and type:

```bash
pip install foodspec
```

**What this does:** Downloads FoodSpec and all the tools it needs to analyze food samples.

**Check it worked:**

```bash
foodspec --version
```

**Expected output:**
```
FoodSpec version 1.0.0
```

If you see an error, verify Python 3.10+ is installed:
```bash
python --version
```

---

## Step 2: Get Sample Data

⏱️ 1 minute

FoodSpec includes example data. Download the oil authentication dataset:

```bash
# Create a working directory
mkdir foodspec-quickstart
cd foodspec-quickstart

# Download sample data (cooking oils Raman spectra)
curl -O https://raw.githubusercontent.com/chandrasekarnarayana/foodspec/main/examples/data/oils.csv
```

**What's in this file?**  
A spreadsheet (CSV) with measurements from 4 types of cooking oils:
- Olive oil (OO)
- Palm oil (PO)
- Sunflower oil (VO)
- Coconut oil (CO)

Each row is a "spectrum" (a measurement showing what molecules are in the oil). The computer can learn patterns to tell them apart.

---

## Step 3: Run Your First Analysis

⏱️ 3 minutes

Run the oil authentication workflow:

```bash
foodspec oil-auth \
  --input oils.csv \
  --output-dir my_first_run
```

**What this command does:**
1. Reads the oil measurements from `oils.csv`
2. Cleans the data (removes noise/background)
3. Trains a computer to recognize each oil type
4. Tests accuracy with cross-validation (checks it's not cheating)
5. Generates a report with charts

**Expected output:**
```yaml
✓ Loaded 120 spectra (30 per oil type)
✓ Preprocessing completed
✓ Classification accuracy: 94.5%
✓ Report saved to: my_first_run/report.html
```

---

## Step 4: View Your Results (5 minutes)

Open the report in your web browser:

```bash
# Mac/Linux
open my_first_run/report.html

# Windows
start my_first_run/report.html

# Or manually open the file in your browser
```

### What You're Looking At

**1. Confusion Matrix (Top Left Chart)**  
A grid showing how often the computer was correct:
- **Diagonal (green):** Correct predictions (e.g., olive oil identified as olive oil)
- **Off-diagonal (red):** Mistakes (e.g., palm oil misclassified as sunflower oil)

**Goal:** All the green should be on the diagonal. If there's red, the computer confused two oils.

**2. Accuracy Number**  
Example: "Balanced Accuracy: 94.5%"  
This means the computer correctly identified the oil type 94.5% of the time.
- **>90% = Excellent** (you can trust this for quality control)
- **70-90% = Good** (usable but may need more data)
- **<70% = Poor** (oils are too similar or data is noisy)

**3. Top Discriminative Features (Bar Chart)**  
Shows which molecular "fingerprints" differ most between oils.
- Higher bars = more important for telling oils apart
- Example: "Ratio 1650/2900" means the ratio of two specific chemical signals

**You don't need to understand the chemistry.** Just know: higher bars = more reliable markers.

**4. Minimal Panel (Bottom Right)**  
The smallest set of measurements needed to identify oils accurately.
- Example: "3 features achieve 92% accuracy"
- **Why this matters:** In a real lab, you'd only measure 3 things instead of 100 (saves time/cost)

---

## Step 5: What Just Happened? (Layer 1 Explanation)

Here's what FoodSpec did behind the scenes, explained like you're 10 years old:

1. **Loaded data:** Opened the spreadsheet with oil measurements
2. **Cleaned it:** Removed background noise (like adjusting TV antenna for clear signal)
3. **Extracted patterns:** Found which signals differ between oils (like comparing fingerprints)
4. **Trained a classifier:** Taught the computer to recognize each oil type (like showing a dog pictures until it learns "cat" vs "dog")
5. **Validated results:** Tested on data it hadn't seen before to make sure it's not memorizing (like a pop quiz)
6. **Made a report:** Generated charts to show you what it learned

**Key insight:** You're not doing chemistry—you're pattern matching with machines. The computer finds differences you can't see by eye.

---

## What's Next?

### Learn More About Data and Terminology

- **[Data Format Reference](../reference/data_format.md)** - How to prepare your own CSV files (schema, validation checklist)
- **[Glossary](../reference/glossary.md)** - What do terms like "wavenumber", "baseline", "CV strategy" mean?
- **[Python Quickstart](quickstart_python.md)** - Run analyses from Python code (for customization)
- **[CLI Quickstart](quickstart_cli.md)** - More detailed command-line examples

### If You Want to Learn More

**Option A: Run Another Example (Beginner)**  
Try the heating quality tutorial to see how oils degrade when fried:
```bash
foodspec heating --help
```
See [thermal_stability_tracking.md](../tutorials/intermediate/03-stability.md).

**Option B: Use Your Own Data (Intermediate)**  
You need a CSV file with:
- One column for sample type (e.g., `oil_type`)
- Columns with measurements (wavenumbers 400-4000 cm⁻¹ for Raman)

See [data_formats_and_hdf5.md](../user-guide/data_formats_and_hdf5.md) for format requirements.

**Option C: Understand the Science (Advanced)**  
Read [spectroscopy_basics.md](../theory/spectroscopy_basics.md) to learn how Raman spectroscopy works.

**Option D: Customize Workflows (Expert)**  
Create your own analysis protocols with YAML:  
See [protocols_and_yaml.md](../user-guide/protocols_and_yaml.md).

---

## Common Problems (and Fixes)

### "Command not found: foodspec"
**Cause:** FoodSpec not installed or not in PATH  
**Fix:**
```bash
pip install --upgrade foodspec
# Verify installation:
pip show foodspec
```

### "File not found: oils.csv"
**Cause:** Didn't download the sample data or wrong directory  
**Fix:**
```bash
# Re-download sample data:
curl -O https://raw.githubusercontent.com/chandrasekarnarayana/foodspec/main/examples/data/oils.csv
# Verify it's there:
ls -l oils.csv
```

### "Accuracy <50%"
**Cause:** Data quality issues or oil types are too similar  
**Fix:**
1. Check your data has at least 20 samples per oil type
2. Verify wavenumber range includes 1600-1800 cm⁻¹ (carbonyl region)
3. See [troubleshooting](../help/troubleshooting.md) for data quality checks

### "No module named 'foodspec'"
**Cause:** Wrong Python environment active  
**Fix:**
```bash
# Check which Python you're using:
which python
python -m pip install foodspec
```

---

## Success Checklist

Before moving to the next tutorial, verify:

- [ ] `foodspec --help` shows command list
- [ ] Sample data downloaded (`ls oils.csv` works)
- [ ] `foodspec oil-auth` completed without errors
- [ ] Report opened in browser showing >90% accuracy
- [ ] Confusion matrix mostly green on diagonal

**All checked?** You're ready for real data! 🎉

---

## FAQ (Absolute Beginner Edition)

**Q: Do I need to know chemistry?**  
A: No. FoodSpec does the chemistry. You just need to run commands and read reports.

**Q: What equipment do I need?**  
A: A Raman or FTIR spectrometer. If you have CSV files of spectra, you already have the data.

**Q: Can I use this for foods other than oils?**  
A: Yes! FoodSpec works on any food with spectroscopy data. Common uses: dairy, honey, spices, beverages.

**Q: Is this suitable for regulatory/legal use?**  
A: FoodSpec provides research-grade tools. For regulatory compliance, see [validation_strategies.md](../05-advanced-topics/validation_strategies.md) for proper validation protocols.

**Q: How do I cite FoodSpec in a paper?**  
A: See the [FAQ](../10-help/faq.md#how-do-i-cite-foodspec) for citation details.

---

## What You've Achieved

✅ Installed FoodSpec  
✅ Ran your first food authentication analysis  
✅ Interpreted a classification report  
✅ Understood the basic workflow (load → clean → analyze → report)

**Next recommended tutorial:** [Oil Discrimination (Basic)](../tutorials/intermediate/01-oil-authentication.md) to learn what each step does in detail.

---

## Need Help?

- **Stuck on an error?** → [Troubleshooting Guide](../10-help/troubleshooting.md) – Installation issues, NaNs, shape errors, and more
- **Have questions?** → [FAQ](../10-help/faq.md) – Baseline methods, sample size, citations
- **Report a bug:** [GitHub Issues](https://github.com/chandrasekarnarayana/foodspec/issues)
