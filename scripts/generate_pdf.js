const path = require('path');
const fs = require('fs');
const { chromium } = require(path.join(__dirname, '..', 'frontend', 'node_modules', '@playwright/test'));

async function generatePDF() {
  console.log("Generating MedClass AI Complete Documentation PDF...");
  
  const templatePath = path.join(__dirname, 'report_template.html');
  const targetPath1 = path.join(__dirname, '..', 'MedClass_AI_Complete_Documentation.pdf');
  const targetPath2 = path.join('C:', 'Users', 'ASUS', 'OneDrive', 'Desktop', 'MedClass_AI_Complete_Documentation.pdf');

  if (!fs.existsSync(templatePath)) {
    throw new Error(`Template not found at: ${templatePath}`);
  }

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();

  const fileUrl = 'file:///' + templatePath.replace(/\\/g, '/');
  console.log(`Loading HTML from: ${fileUrl}`);
  await page.goto(fileUrl, { waitUntil: 'networkidle' });

  // Generate PDF with high quality
  await page.pdf({
    path: targetPath1,
    format: 'A4',
    printBackground: true,
    margin: {
      top: '18mm',
      bottom: '18mm',
      left: '14mm',
      right: '14mm'
    }
  });
  console.log(`Successfully generated PDF at: ${targetPath1}`);

  // Copy to Desktop if available
  try {
    fs.copyFileSync(targetPath1, targetPath2);
    console.log(`Successfully copied PDF to Desktop: ${targetPath2}`);
  } catch (err) {
    console.log(`Note: Could not copy to Desktop directly (${err.message}), primary PDF is at: ${targetPath1}`);
  }

  await browser.close();
  console.log("PDF generation process complete!");
}

generatePDF().catch(err => {
  console.error("PDF generation failed:", err);
  process.exit(1);
});
