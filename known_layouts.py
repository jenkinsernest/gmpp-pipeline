"""Known raw GMPP header layouts, 2021-22 to 2025-26.

These fingerprints were taken from the stored raw files in gmpp-data,
which match the official GOV.UK CSV attachments byte for byte (checked
12 November 2026). A new file is accepted automatically only if its
parsed column names match one of these lists, or a layout the user has
approved with --approve-layout.
"""

from __future__ import annotations

KNOWN_LAYOUTS = {
    '2021-22': {
        'year': '2021-22',
        'notes': '17 columns. Project ID is first and is called GMPP ID Number. No benefits columns. IPA and SRO titles include the long three-point scale definition.',
        'columns': [
            'GMPP ID Number',
            'Project Name',
            'Department',
            'Annual Report Category',
            'Description / Aims',
            'IPA Delivery Confidence Assessment \n(A Delivery Confidence Assessment of the project at a fixed point in time, using a three-point scale, Red – Amber – Green; definitions in the IPA Annual Report on Major Projects)',
            'SRO Delivery Confidence Assessment \n(A Delivery Confidence Assessment of the project at a fixed point in time, using a three-point scale, Red – Amber – Green; definitions in the IPA Annual Report on Major Projects)',
            'Departmental commentary on actions planned or taken on the IPA RAG rating.',
            'Project - Start Date \n(Latest Approved Start Date)',
            'Project - End Date \n(Latest Approved End Date)',
            'Departmental narrative on schedule, including any deviation from planned schedule (if necessary)',
            'Financial Year Baseline (£m) (including Non-Government Costs)',
            'Financial Year Forecast (£m) (including Non-Government Costs)',
            'Financial Year Variance (%)',
            'Departmental narrative on  budget/forecast variance for 2021/22 \n(if variance is more than 5%)',
            'TOTAL Baseline Whole Life Costs (£m) (including Non-Government Costs)',
            'Departmental Narrative on Budgeted Whole Life Costs',
        ],
    },
    '2022-23': {
        'year': '2022-23',
        'notes': '19 columns. Same titles as 2021-22, including the variance narrative still labelled for 2021/22, plus total baseline benefits and a benefits narrative.',
        'columns': [
            'GMPP ID Number',
            'Project Name',
            'Department',
            'Annual Report Category',
            'Description / Aims',
            'IPA Delivery Confidence Assessment \n(A Delivery Confidence Assessment of the project at a fixed point in time, using a three-point scale, Red – Amber – Green; definitions in the IPA Annual Report on Major Projects)',
            'SRO Delivery Confidence Assessment \n(A Delivery Confidence Assessment of the project at a fixed point in time, using a three-point scale, Red – Amber – Green; definitions in the IPA Annual Report on Major Projects)',
            'Departmental commentary on actions planned or taken on the IPA RAG rating.',
            'Project - Start Date \n(Latest Approved Start Date)',
            'Project - End Date \n(Latest Approved End Date)',
            'Departmental narrative on schedule, including any deviation from planned schedule (if necessary)',
            'Financial Year Baseline (£m) (including Non-Government Costs)',
            'Financial Year Forecast (£m) (including Non-Government Costs)',
            'Financial Year Variance (%)',
            'Departmental narrative on  budget/forecast variance for 2021/22 \n(if variance is more than 5%)',
            'TOTAL Baseline Whole Life Costs (£m) (including Non-Government Costs)',
            'Departmental Narrative on Budgeted Whole Life Costs',
            'TOTAL Baseline Benefits (£m)',
            'Departmental Narrative on Budgeted Benefits',
        ],
    },
    '2023-24': {
        'year': '2023-24',
        'notes': "19 columns. Same set as 2022-23, except the variance narrative title is updated to 2023/24 and the extra space after 'on' is removed.",
        'columns': [
            'GMPP ID Number',
            'Project Name',
            'Department',
            'Annual Report Category',
            'Description / Aims',
            'IPA Delivery Confidence Assessment \n(A Delivery Confidence Assessment of the project at a fixed point in time, using a three-point scale, Red – Amber – Green; definitions in the IPA Annual Report on Major Projects)',
            'SRO Delivery Confidence Assessment \n(A Delivery Confidence Assessment of the project at a fixed point in time, using a three-point scale, Red – Amber – Green; definitions in the IPA Annual Report on Major Projects)',
            'Departmental commentary on actions planned or taken on the IPA RAG rating.',
            'Project - Start Date \n(Latest Approved Start Date)',
            'Project - End Date \n(Latest Approved End Date)',
            'Departmental narrative on schedule, including any deviation from planned schedule (if necessary)',
            'Financial Year Baseline (£m) (including Non-Government Costs)',
            'Financial Year Forecast (£m) (including Non-Government Costs)',
            'Financial Year Variance (%)',
            'Departmental narrative on budget/forecast variance for 2023/24 \n(if variance is more than 5%)',
            'TOTAL Baseline Whole Life Costs (£m) (including Non-Government Costs)',
            'Departmental Narrative on Budgeted Whole Life Costs',
            'TOTAL Baseline Benefits (£m)',
            'Departmental Narrative on Budgeted Benefits',
        ],
    },
    '2024-25': {
        'year': '2024-25',
        'notes': '21 columns. This is the big redesign. Titles are shortened, the ID moves to the last column and is renamed GMPP ID, and two new columns appear: the evaluation plan question and the SRO name.',
        'columns': [
            'Project Name',
            'Department',
            'Annual Report Category',
            'Project Description',
            'IPA Delivery Confidence Assessment',
            'SRO Delivery Confidence Assessment',
            'Departmental Commentary on Delivery Confidence Assessment Rating',
            'Start Date',
            'End Date',
            'Schedule Narrative',
            'Financial Year Baseline (£m)',
            'Financial Year Forecast (£m)',
            'Financial Year Variance (%)',
            'In Year Variance Narrative',
            'Whole Life Cost (£m)',
            'Costs Narrative',
            'Benefits (£m)',
            'Benefits Narrative',
            'Does the project have an evaluation plan? (self reported)',
            'Senior Responsible Owner (SRO) Name',
            'GMPP ID',
        ],
    },
    '2025-26': {
        'year': '2025-26',
        'notes': "21 columns, same order as 2024-25. Cost and benefit titles add 'Presented in 2024/25 Real Prices'. The evaluation column becomes Evaluation Plan and Registry Link. The ID is renamed Major Projects ID.",
        'columns': [
            'Project Name',
            'Department',
            'Annual Report Category',
            'Project Description',
            'IPA Delivery Confidence Assessment',
            'SRO Delivery Confidence Assessment',
            'Departmental Commentary on Delivery Confidence Assessment Rating',
            'Start Date',
            'End Date',
            'Schedule Narrative',
            'Financial Year Baseline (£m, Presented in 2024/25 Real Prices)',
            'Financial Year Forecast (£m, Presented in 2024/25 Real Prices)',
            'Financial Year Variance (%)',
            'In Year Variance Narrative',
            'Whole Life Cost (£m, Presented in 2024/25 Real Prices)',
            'Costs Narrative',
            'Benefits (£m, Presented in 2024/25 Real Prices)',
            'Benefits Narrative',
            'Evaluation Plan and Registry Link',
            'Senior Responsible Owner (SRO) Name',
            'Major Projects ID',
        ],
    },
}

CLEANED_COLUMNS = [
    'Major Projects ID',
    'Department',
    'Annual Report Category',
    'Whole Life Cost',
    'Financial Year Baseline',
    'Financial Year Forecast',
    'Benefits',
    'Start Date',
    'End Date',
    'IPA Delivery Confidence',
]

def normalize_header(name: str) -> str:
    """Collapse line breaks and repeated spaces, but keep punctuation and case."""
    return " ".join(str(name).replace("\r", " ").replace("\n", " ").split())


def normalized_columns(columns) -> list[str]:
    return [normalize_header(c) for c in columns]


def describe_known_layout(year: str) -> str:
    item = KNOWN_LAYOUTS[year]
    return f"{year} ({len(item['columns'])} columns). {item['notes']}"


def match_known_layout(columns, extra_layouts=None):
    """Return (status, label, detail).

    status is 'known', 'approved', or 'unknown'.
    """
    got = normalized_columns(columns)
    for year, item in KNOWN_LAYOUTS.items():
        if normalized_columns(item["columns"]) == got:
            return "known", year, item["notes"]
    for item in extra_layouts or []:
        if normalized_columns(item.get("columns") or []) == got:
            return "approved", item.get("id") or "approved", item.get("notes") or "User-approved layout"
    return "unknown", None, "Column names do not match any known or approved layout."
