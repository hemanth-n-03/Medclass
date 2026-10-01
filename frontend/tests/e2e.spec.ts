import { test, expect } from "@playwright/test";

test.describe("MedClass AI End-to-End Test Suite", () => {
  
  test("1. Landing page loads cleanly with hero and navigation", async ({ page }) => {
    await page.goto("/");
    await expect(page).toHaveTitle(/MedClass AI/);
    await expect(page.locator("h1")).toContainText("Understand Medical Questions");
    await expect(page.getByRole("link", { name: "Try Classifier" }).first()).toBeVisible();
    await expect(page.getByRole("link", { name: "Explore Analytics" })).toBeVisible();
  });

  test("2. Full clinical question classification flow", async ({ page }) => {
    await page.goto("/classify");
    await expect(page.locator("h1")).toContainText("Medical Question Classifier");

    // Fill query
    const textarea = page.locator("textarea");
    await textarea.fill("What are the early symptoms of diabetes?");
    await page.getByRole("button", { name: "Classify Question" }).click();

    // Verify prediction results appear
    await expect(page.locator("text=Medical Specialty Category")).toBeVisible({ timeout: 15000 });
    await expect(page.getByRole("heading", { name: "Endocrinology" })).toBeVisible();
    await expect(page.locator("text=Clinical Intention")).toBeVisible();
    await expect(page.getByRole("heading", { name: "Symptoms" })).toBeVisible();

    // Verify similar questions and safety notice
    await expect(page.locator("text=Similar Questions from Verified NIH Knowledge Base")).toBeVisible();
    await expect(page.locator("text=This system classifies medical questions and is not a medical diagnostic tool")).toBeVisible();
  });

  test("3. Non-medical out-of-domain query handling", async ({ page }) => {
    await page.goto("/classify");
    const textarea = page.locator("textarea");
    await textarea.fill("How do I repair my laptop battery?");
    await page.getByRole("button", { name: "Classify Question" }).click();

    // Verify out-of-domain notification
    await expect(page.locator("text=Outside Supported Medical Domain")).toBeVisible({ timeout: 15000 });
  });

  test("4. Emergency red flag symptom triage trigger", async ({ page }) => {
    await page.goto("/classify");
    const textarea = page.locator("textarea");
    await textarea.fill("I have severe crushing chest pain radiating to my left arm");
    await page.getByRole("button", { name: "Classify Question" }).click();

    // Verify critical emergency alert
    await expect(page.locator("text=Critical Emergency Symptom Detected")).toBeVisible({ timeout: 15000 });
  });

  test("5. Empty input validation prevention", async ({ page }) => {
    await page.goto("/classify");
    const classifyBtn = page.getByRole("button", { name: "Classify Question" });
    await expect(classifyBtn).toBeDisabled();
  });

  test("6. Analytics dashboard visualization", async ({ page }) => {
    await page.goto("/analytics");
    await expect(page.locator("h1")).toContainText("Analytics & Model Intelligence");
    await expect(page.locator("text=Total Inquiries")).toBeVisible();
    await expect(page.locator("text=Clinical Queries")).toBeVisible();
  });

  test("7. Model Architecture and empirical benchmark specs", async ({ page }) => {
    await page.goto("/models");
    await expect(page.locator("h1")).toContainText("Model Engineering & Benchmarks");
    await expect(page.locator("text=Empirical Benchmark Results")).toBeVisible();
    await expect(page.locator("text=Baseline: TF-IDF + Linear SVM")).toBeVisible();
    await expect(page.locator("text=Transformer SOTA: Multi-Task BERT")).toBeVisible();
  });

  test("8. Responsible AI, Ethics and Safety page", async ({ page }) => {
    await page.goto("/about");
    await expect(page.locator("h1")).toContainText("About MedClass AI & Safety Ethics");
    await expect(page.locator("text=What the System DOES")).toBeVisible();
    await expect(page.locator("text=What the System NEVER Does")).toBeVisible();
    await expect(page.locator("text=Never Diagnoses")).toBeVisible();
  });

});
