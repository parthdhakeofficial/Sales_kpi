# report_pdf.py

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
    Image
)

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4


class SalesKPIReport:

    def __init__(self, output_path="reports/Sales_KPI_Report.pdf"):

        self.output_path = output_path
        self.styles = getSampleStyleSheet()

        self.title_style = self.styles["Title"]
        self.heading_style = self.styles["Heading2"]
        self.body_style = self.styles["BodyText"]

    def create_kpi_table(self, kpi_dict):

        data = [["KPI", "Value"]]

        for k, v in kpi_dict.items():
            data.append([str(k), str(v)])

        table = Table(data, colWidths=[220, 220])

        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke)
        ]))

        return table

    def create_dataframe_table(self, dataframe):

        table_data = [list(dataframe.columns)]

        for row in dataframe.values.tolist():
            table_data.append(row)

        table = Table(table_data)

        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
        ]))

        return table

    def add_chart(self, story, image_path, title):

        try:

            story.append(
                Paragraph(
                    title,
                    self.heading_style
                )
            )

            story.append(
                Image(
                    image_path,
                    width=500,
                    height=250
                )
            )

            story.append(
                Spacer(1, 10)
            )

        except Exception:

            story.append(
                Paragraph(
                    f"Chart not found: {image_path}",
                    self.body_style
                )
            )

    def generate(
            self,
            kpis,
            executive_summary,
            revenue_pivot,
            region_pivot,
            product_pivot,
            ml_insights,
            anomalies,
            recommendations,
            forecast_summary,
            chart_paths
    ):

        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=A4
        )

        story = []

        # ==================================================
        # COVER PAGE
        # ==================================================

        story.append(
            Paragraph(
                "Sales KPI Executive Report",
                self.title_style
            )
        )

        story.append(Spacer(1, 20))

        story.append(
            Paragraph(
                "Board Level Business Performance Report",
                self.body_style
            )
        )

        story.append(PageBreak())

        # ==================================================
        # EXECUTIVE SUMMARY
        # ==================================================

        story.append(
            Paragraph(
                "1. Executive Summary",
                self.heading_style
            )
        )

        story.append(
            Paragraph(
                executive_summary,
                self.body_style
            )
        )

        story.append(Spacer(1, 20))

        # ==================================================
        # KPI SNAPSHOT
        # ==================================================

        story.append(
            Paragraph(
                "2. KPI Snapshot",
                self.heading_style
            )
        )

        story.append(
            self.create_kpi_table(kpis)
        )

        story.append(PageBreak())

        # ==================================================
        # REVENUE ANALYSIS
        # ==================================================

        story.append(
            Paragraph(
                "3. Revenue Analysis",
                self.heading_style
            )
        )

        self.add_chart(
            story,
            chart_paths["revenue"],
            "Revenue Trend"
        )

        story.append(PageBreak())

        # ==================================================
        # SALES ANALYSIS
        # ==================================================

        story.append(
            Paragraph(
                "4. Sales Analysis",
                self.heading_style
            )
        )

        self.add_chart(
            story,
            chart_paths["sales"],
            "Sales Trend"
        )

        story.append(PageBreak())

        # ==================================================
        # PROFITABILITY
        # ==================================================

        story.append(
            Paragraph(
                "5. Profitability Analysis",
                self.heading_style
            )
        )

        self.add_chart(
            story,
            chart_paths["profit"],
            "Profit Trend"
        )

        story.append(PageBreak())

        # ==================================================
        # REVENUE PIVOT
        # ==================================================

        story.append(
            Paragraph(
                "6. Revenue Pivot Analysis",
                self.heading_style
            )
        )

        story.append(
            self.create_dataframe_table(
                revenue_pivot
            )
        )

        story.append(PageBreak())

        # ==================================================
        # REGION PIVOT
        # ==================================================

        story.append(
            Paragraph(
                "7. Region Performance",
                self.heading_style
            )
        )

        story.append(
            self.create_dataframe_table(
                region_pivot
            )
        )

        story.append(PageBreak())

        # ==================================================
        # PRODUCT PIVOT
        # ==================================================

        story.append(
            Paragraph(
                "8. Product Analysis",
                self.heading_style
            )
        )

        story.append(
            self.create_dataframe_table(
                product_pivot
            )
        )

        story.append(PageBreak())

        # ==================================================
        # ML INSIGHTS
        # ==================================================

        story.append(
            Paragraph(
                "9. Machine Learning Insights",
                self.heading_style
            )
        )

        for item in ml_insights:

            text = f"""
            <b>Observation:</b> {item['observation']}<br/>
            <b>Impact:</b> {item['impact']}<br/>
            <b>Cause:</b> {item['cause']}<br/>
            <b>Recommendation:</b> {item['recommendation']}<br/>
            <b>Confidence:</b> {item['confidence']}%
            """

            story.append(
                Paragraph(
                    text,
                    self.body_style
                )
            )

            story.append(
                Spacer(1, 10)
            )

        story.append(PageBreak())

        # ==================================================
        # ANOMALIES
        # ==================================================

        story.append(
            Paragraph(
                "10. Anomaly Detection Findings",
                self.heading_style
            )
        )

        for anomaly in anomalies:

            story.append(
                Paragraph(
                    f"• {anomaly}",
                    self.body_style
                )
            )

        story.append(PageBreak())

        # ==================================================
        # FORECAST
        # ==================================================

        story.append(
            Paragraph(
                "11. Forecast & Outlook",
                self.heading_style
            )
        )

        story.append(
            Paragraph(
                forecast_summary,
                self.body_style
            )
        )

        self.add_chart(
            story,
            chart_paths["forecast"],
            "Revenue Forecast"
        )

        story.append(PageBreak())

        # ==================================================
        # RECOMMENDATIONS
        # ==================================================

        story.append(
            Paragraph(
                "12. Executive Recommendations",
                self.heading_style
            )
        )

        for rec in recommendations:

            text = f"""
            <b>Priority:</b> {rec['priority']}<br/>
            <b>Action:</b> {rec['action']}<br/>
            <b>Expected Benefit:</b> {rec['benefit']}<br/>
            <b>Risk:</b> {rec['risk']}
            """

            story.append(
                Paragraph(
                    text,
                    self.body_style
                )
            )

            story.append(
                Spacer(1, 10)
            )

        doc.build(story)

        print(
            f"Report generated successfully -> "
            f"{self.output_path}"
        )