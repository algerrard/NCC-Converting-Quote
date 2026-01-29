import streamlit as st
import pandas as pd
import numpy as np
from azure.storage.blob import BlobServiceClient
from io import StringIO, BytesIO
from datetime import datetime
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="NCC Converting Quote", page_icon="📦", layout="wide")

# Session state initialization
if "quote_params" not in st.session_state:
    st.session_state.quote_params = {}
if "quote_result" not in st.session_state:
    st.session_state.quote_result = None

# Secrets
try:
    AZURE_CONNECTION_STRING = os.environ.get("AZURE_CONNECTION_STRING")
    if not AZURE_CONNECTION_STRING:
        AZURE_CONNECTION_STRING = st.secrets["AZURE_CONNECTION_STRING"]
    if not AZURE_CONNECTION_STRING:
        raise KeyError("AZURE_CONNECTION_STRING")
except KeyError as e:
    st.error(f"Missing secret: {e}")
    st.stop()

CONTAINER_NAME = "data"
PAPER_INFO_BLOB = "PaperInfoNCC.csv"
MACHINE_INFO_BLOB = "MachineInfo.csv"

# =========================================================
# DATA LOADING
# =========================================================
@st.cache_data(ttl=3600)
def load_paper_info():
    """Load PaperInfoNCC.csv from Azure Blob Storage."""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        paper_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=PAPER_INFO_BLOB)
        paper_csv = paper_client.download_blob().readall().decode("utf-8")
        paper_df = pd.read_csv(StringIO(paper_csv))
        paper_df.columns = paper_df.columns.str.strip()
        return paper_df
    except Exception as e:
        st.error(f"Error loading PaperInfoNCC: {str(e)}")
        return None


@st.cache_data(ttl=3600)
def load_machine_info():
    """Load MachineInfo.csv from Azure Blob Storage."""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        machine_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=MACHINE_INFO_BLOB)
        machine_csv = machine_client.download_blob().readall().decode("utf-8")
        machine_df = pd.read_csv(StringIO(machine_csv))
        machine_df.columns = machine_df.columns.str.strip()
        return machine_df
    except Exception as e:
        st.error(f"Error loading MachineInfo: {str(e)}")
        return None


# =========================================================
# CALCULATION FUNCTIONS
# =========================================================
def clean_currency(value, default=0):
    """
    Clean a currency string (e.g., '$273.00') to a float.
    """
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    # Remove $, commas, and whitespace
    cleaned = str(value).replace('$', '').replace(',', '').strip()
    try:
        return float(cleaned)
    except ValueError:
        return default


def calculate_roll_weight(diameter, core, width, density_factor):
    """
    Calculate the approximate weight of a roll.
    Weight = (Diameter² - Core²) × Width × Density_factor
    """
    return (diameter**2 - core**2) * width * density_factor


def calculate_base_rate(params, paper_df, machine_df, product_group_col="Product Group"):
    """
    Calculate the base converting rate in $/CWT.

    Returns dict with calculation details and final rate.
    """
    result = {
        "success": False,
        "error": None,
        "details": {},
        "base_rate_cwt": None
    }

    try:
        # Extract parameters
        product_group = params.get("product_group")
        basis_weight = params.get("basis_weight")
        basis_weight_unit = params.get("basis_weight_unit", "LBS")
        caliper = params.get("caliper", 0)
        cut_width = params.get("cut_width")
        parent_roll_width = params.get("parent_roll_width")
        parent_roll_diameter = params.get("parent_roll_diameter")
        parent_roll_core = params.get("parent_roll_core", 3.0)
        quantity_lbs = params.get("quantity_lbs")
        service_type = params.get("service_type")  # "Rewinder", "Sheeter", or "Both"

        # Lookup paper info by Product Group
        paper_row = paper_df[paper_df[product_group_col].astype(str).str.strip() == str(product_group).strip()]
        if paper_row.empty:
            result["error"] = f"Product Group '{product_group}' not found in PaperInfoNCC"
            return result
        paper_row = paper_row.iloc[0]

        # Get paper parameters
        area_in = float(paper_row.get("Area(IN)", 0) or 0)
        density_factor = float(paper_row.get("Density_Factor", 0) or 0)
        gsm_factor = float(paper_row.get("GSM_Factor", 3100) or 3100)
        run_adjust = float(paper_row.get("RunAdjust", 1.0) or 1.0)
        num_shtr_rolls = int(paper_row.get("NumShtrRolls", 1) or 1)

        if area_in == 0:
            result["error"] = "Area(IN) is missing or zero in PaperInfoNCC"
            return result
        if density_factor == 0:
            result["error"] = "Density_Factor is missing or zero in PaperInfoNCC"
            return result

        # Convert basis weight to LBS if needed
        if basis_weight_unit == "GSM":
            basis_weight_lbs = basis_weight / gsm_factor
        else:
            basis_weight_lbs = basis_weight

        # Determine equipment type for machine lookup
        if service_type == "Both":
            equip_type = "Sheeter"  # Use sheeter rates when both are selected
        else:
            equip_type = service_type

        # Lookup machine info
        machine_row = machine_df[machine_df["EquipType"].astype(str).str.strip() == equip_type]
        if machine_row.empty:
            result["error"] = f"Equipment type '{equip_type}' not found in MachineInfo"
            return result
        machine_row = machine_row.iloc[0]

        # Get machine parameters
        avg_speed = float(machine_row.get("AvgSpeed(FPM)", machine_row.get("avgspeed(FPM)", machine_row.get("AvgSpeed", 2200))) or 2200)
        hourly_rate = clean_currency(machine_row.get("HourlyRate"), 273)
        roll_change_hrs = float(machine_row.get("Roll_Change_Hrs", 0.25) or 0.25)
        setup_hrs = float(machine_row.get("Setup_Hrs", 0.5) or 0.5)

        # Step 1: Calculate average roll weight
        avg_roll_weight = calculate_roll_weight(
            parent_roll_diameter,
            parent_roll_core,
            parent_roll_width,
            density_factor
        )

        # Step 2: Calculate number of rolls (round up - no partial rolls)
        num_rolls = int(np.ceil(quantity_lbs / avg_roll_weight)) if avg_roll_weight > 0 else 1
        num_rolls = max(1, num_rolls)  # At least 1 roll

        # Step 2b: Determine rolls running at one time (for sheeting)
        # If sheeting and caliper <= 11 (or blank/zero), use NumShtrRolls
        # Otherwise, 1 roll at a time
        if equip_type == "Sheeter" and (caliper == 0 or caliper is None or caliper <= 11):
            rolls_running = num_shtr_rolls
        else:
            rolls_running = 1

        # Step 3: Calculate LbsPerHour
        # LbsPerHour = (BasisWt / (Area × 500)) × Cut Width × (AvgSpeed × 12) × 60 × RunAdjust
        # For sheeting with multiple rolls, multiply by rolls_running
        lbs_per_hour = (
            (basis_weight_lbs / (area_in * 500))
            * cut_width
            * (avg_speed * 12)
            * 60
            * run_adjust
            * rolls_running
        )

        # Step 4: Calculate processing hours
        processing_hours = quantity_lbs / lbs_per_hour if lbs_per_hour > 0 else 0

        # Step 5: Calculate roll change hours
        # Rewinding: roll changes = number of parent rolls × Roll_Change_Hrs
        # Sheeting: roll changes = number of sets (total rolls / rolls running) × Roll_Change_Hrs
        if equip_type == "Rewinder":
            num_roll_changes = num_rolls
        else:
            # Sheeting: number of sets
            num_roll_changes = int(np.ceil(num_rolls / rolls_running)) if rolls_running > 0 else num_rolls
        roll_change_hours = roll_change_hrs * num_roll_changes

        # Step 6: Total hours (processing + roll changes + setup)
        total_hours = processing_hours + roll_change_hours + setup_hrs

        # Step 7: Total cost
        total_cost = total_hours * hourly_rate

        # Step 8: $/CWT
        base_rate_cwt = (total_cost / quantity_lbs) * 100 if quantity_lbs > 0 else 0

        # Populate result
        result["success"] = True
        result["base_rate_cwt"] = round(base_rate_cwt, 2)
        result["details"] = {
            "product_group": product_group,
            "basis_weight_lbs": round(basis_weight_lbs, 4),
            "caliper": caliper,
            "area_in": area_in,
            "density_factor": density_factor,
            "run_adjust": run_adjust,
            "num_shtr_rolls": num_shtr_rolls,
            "avg_speed_fpm": avg_speed,
            "hourly_rate": hourly_rate,
            "roll_change_hrs": roll_change_hrs,
            "setup_hrs": setup_hrs,
            "avg_roll_weight": round(avg_roll_weight, 2),
            "num_rolls": num_rolls,
            "rolls_running": rolls_running,
            "num_roll_changes": num_roll_changes,
            "lbs_per_hour": round(lbs_per_hour, 2),
            "processing_hours": round(processing_hours, 4),
            "roll_change_hours": round(roll_change_hours, 4),
            "total_hours": round(total_hours, 4),
            "total_cost": round(total_cost, 2),
            "quantity_lbs": quantity_lbs,
            "service_type": service_type,
            "equip_type": equip_type
        }

        return result

    except Exception as e:
        result["error"] = str(e)
        return result


# =========================================================
# MAIN APP
# =========================================================
def main():
    st.title("NCC Converting Quote")

    # Load data
    paper_df = load_paper_info()
    machine_df = load_machine_info()

    if paper_df is None or machine_df is None:
        st.error("Failed to load required data. Please check Azure connection.")
        st.stop()

    # Debug: Show available columns (remove after identifying correct column)
    with st.expander("Debug: PaperInfoNCC Columns"):
        st.write("Available columns:", paper_df.columns.tolist())
        st.write("First few rows:")
        st.dataframe(paper_df.head())

    # Get unique Product Groups for dropdown
    # Try to find the Product Group column (may have different name)
    product_group_col = None
    for col in paper_df.columns:
        if "product" in col.lower() and "group" in col.lower():
            product_group_col = col
            break

    if product_group_col is None:
        st.error(f"Could not find 'Product Group' column. Available columns: {paper_df.columns.tolist()}")
        st.stop()

    product_groups = paper_df[product_group_col].dropna().unique().tolist()
    product_groups.sort()

    # =========================================================
    # INPUT SECTION
    # =========================================================
    st.header("Quote Parameters")

    # Service Selection with larger font
    st.subheader("Service Type")
    st.markdown("""
        <style>
        div[data-testid="stRadio"] > label {
            font-size: 1.2rem !important;
            font-weight: 500;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label {
            font-size: 1.1rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

    service_selection = st.radio(
        "Select service type:",
        options=["Rewinding", "Sheeting"],
        horizontal=True,
        label_visibility="collapsed"
    )

    # Determine service type
    if service_selection == "Rewinding":
        service_type = "Rewinder"
    else:
        service_type = "Sheeter"

    st.divider()

    # Parameters
    st.subheader("Material & Job Specifications")

    col1, col2 = st.columns(2)

    with col1:
        product_group = st.selectbox(
            "Product Group",
            options=product_groups,
            index=0 if product_groups else None
        )

        # Basis Weight with unit toggle
        basis_weight_unit = st.radio(
            "Basis Weight Unit",
            options=["LBS", "GSM"],
            horizontal=True
        )
        basis_weight = st.number_input(
            f"Basis Weight ({basis_weight_unit})",
            min_value=0.0,
            value=0.0,
            step=0.1,
            format="%.2f"
        )

        caliper = st.number_input(
            "Caliper (PTS)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            format="%.2f"
        )

        cut_width = st.number_input(
            "Cut Width Total (IN)",
            min_value=0.0,
            value=0.0,
            step=0.25,
            format="%.2f"
        )

        sheet_length = st.number_input(
            "Sheet Length (IN)",
            min_value=0.0,
            value=0.0,
            step=0.25,
            format="%.2f"
        )

    with col2:
        parent_roll_width = st.number_input(
            "Parent Roll Width (IN)",
            min_value=0.0,
            value=0.0,
            step=0.25,
            format="%.2f"
        )

        parent_roll_diameter = st.number_input(
            "Parent Roll Diameter (IN)",
            min_value=0.0,
            value=0.0,
            step=0.5,
            format="%.2f"
        )

        parent_roll_core = st.number_input(
            "Parent Roll Core (IN)",
            min_value=0.0,
            value=3.0,
            step=0.5,
            format="%.2f",
            help="Default is 3 inches if not specified"
        )

        quantity_lbs = st.number_input(
            "Quantity - Raw Material (LBS)",
            min_value=0.0,
            value=0.0,
            step=100.0,
            format="%.0f"
        )

    st.divider()

    # =========================================================
    # CALCULATE QUOTE
    # =========================================================
    if st.button("Calculate Quote", type="primary", use_container_width=True):
        # Validate inputs
        errors = []
        if basis_weight <= 0:
            errors.append("Basis Weight must be greater than 0")
        if cut_width <= 0:
            errors.append("Cut Width Total must be greater than 0")
        if parent_roll_width <= 0:
            errors.append("Parent Roll Width must be greater than 0")
        if parent_roll_diameter <= 0:
            errors.append("Parent Roll Diameter must be greater than 0")
        if parent_roll_core >= parent_roll_diameter:
            errors.append("Parent Roll Core must be less than Parent Roll Diameter")
        if quantity_lbs <= 0:
            errors.append("Quantity must be greater than 0")

        if errors:
            for error in errors:
                st.error(error)
        else:
            # Build params
            params = {
                "product_group": product_group,
                "basis_weight": basis_weight,
                "basis_weight_unit": basis_weight_unit,
                "caliper": caliper,
                "cut_width": cut_width,
                "sheet_length": sheet_length,
                "parent_roll_width": parent_roll_width,
                "parent_roll_diameter": parent_roll_diameter,
                "parent_roll_core": parent_roll_core if parent_roll_core > 0 else 3.0,
                "quantity_lbs": quantity_lbs,
                "service_type": service_type
            }

            # Calculate
            result = calculate_base_rate(params, paper_df, machine_df, product_group_col)
            st.session_state.quote_result = result
            st.session_state.quote_params = params

    # =========================================================
    # DISPLAY RESULTS
    # =========================================================
    if st.session_state.quote_result is not None:
        result = st.session_state.quote_result

        st.header("Quote Result")

        if result["success"]:
            # Display main result prominently
            st.metric(
                label="Base Rate",
                value=f"${result['base_rate_cwt']:.2f} / CWT"
            )

            # Show calculation details in expander
            with st.expander("View Calculation Details"):
                details = result["details"]

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Input Parameters**")
                    st.write(f"- Product Group: {details['product_group']}")
                    st.write(f"- Service Type: {details['service_type']}")
                    st.write(f"- Basis Weight (LBS): {details['basis_weight_lbs']}")
                    st.write(f"- Quantity: {details['quantity_lbs']:,.0f} lbs")

                    st.markdown("**Paper Information**")
                    st.write(f"- Area (IN): {details['area_in']}")
                    st.write(f"- Density Factor: {details['density_factor']}")
                    st.write(f"- Run Adjust: {details['run_adjust']}")

                with col2:
                    st.markdown("**Machine Information**")
                    st.write(f"- Equipment Type: {details['equip_type']}")
                    st.write(f"- Avg Speed (FPM): {details['avg_speed_fpm']}")
                    st.write(f"- Hourly Rate: ${details['hourly_rate']:.2f}")
                    st.write(f"- Roll Change (hrs): {details['roll_change_hrs']}")

                    st.markdown("**Calculated Values**")
                    st.write(f"- Avg Roll Weight: {details['avg_roll_weight']:,.2f} lbs")
                    st.write(f"- Number of Rolls: {details['num_rolls']:.2f}")
                    st.write(f"- Lbs Per Hour: {details['lbs_per_hour']:,.2f}")
                    st.write(f"- Processing Hours: {details['processing_hours']:.4f}")
                    st.write(f"- Roll Change Hours: {details['roll_change_hours']:.4f}")
                    st.write(f"- Total Hours: {details['total_hours']:.4f}")
                    st.write(f"- Total Cost: ${details['total_cost']:.2f}")
        else:
            st.error(f"Calculation Error: {result['error']}")


if __name__ == "__main__":
    main()
