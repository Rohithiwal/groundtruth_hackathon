from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

def create_store_policy():
    filename = "store_policy.pdf"
    
    # Remove old file if exists to prevent corruption
    if os.path.exists(filename):
        os.remove(filename)

    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Groundtruth Store Policy & Partnerships (2025)")

    # Body Text
    c.setFont("Helvetica", 12)
    text_lines = [
        "",
        "1. RETURN & REFUND POLICY",
        "- Standard Window: Customers have 30 days from purchase to return unworn items.",
        "- Holiday Extension: Items purchased between Nov 1 and Dec 31 can be returned until Jan 31.",
        "- 'No Questions Asked': Applies ONLY to Gold Tier members.",
        "",
        "2. LOCATION-BASED PARTNERSHIPS",
        "- Starbucks Partnership: Any customer detected within 50 meters of a Starbucks branch",
        "  is eligible for a 10% discount coupon. Code: COFFEE10.",
        "- Movie Theater Cross-Promo: Customers near 'AMC Theaters' get a BOGO sock voucher.",
        "",
        "3. INVENTORY ALERTS",
        "- Winter Jackets: Low stock in Downtown branch. Recommend 'Ship to Home'.",
        "- Thermal Wear: Overstocked. Push 20% discounts for 'cold' queries.",
    ]

    y_position = height - 80
    for line in text_lines:
        c.drawString(50, y_position, line)
        y_position -= 20

    c.save()
    print(f"✅ Successfully created {filename}")

if __name__ == "__main__":
    create_store_policy()