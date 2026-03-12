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
if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0

# Secrets — check env var first (Azure App Service), then st.secrets (Streamlit Cloud)
AZURE_CONNECTION_STRING = os.environ.get("AZURE_CONNECTION_STRING")
if not AZURE_CONNECTION_STRING:
    try:
        AZURE_CONNECTION_STRING = st.secrets["AZURE_CONNECTION_STRING"]
    except Exception:
        AZURE_CONNECTION_STRING = None
if not AZURE_CONNECTION_STRING:
    st.error("Missing AZURE_CONNECTION_STRING. Set as env var or in .streamlit/secrets.toml")
    st.stop()

CONTAINER_NAME = "data"
PAPER_INFO_BLOB = "PaperInfoNCC.csv"
MACHINE_INFO_BLOB = "MachineInfo.csv"
ADD_CHARGE_BLOB = "NCC Add Charge Schedule.csv"
ORDER_SIZE_BLOB = "Order Size Adjustments.csv"

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


@st.cache_data(ttl=3600)
def load_add_charges():
    """Load NCC Add Charge Schedule.csv from Azure Blob Storage."""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=ADD_CHARGE_BLOB)
        csv_data = blob_client.download_blob().readall().decode("utf-8")
        df = pd.read_csv(StringIO(csv_data))
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading Add Charge Schedule: {str(e)}")
        return None


@st.cache_data(ttl=3600)
def load_order_size_adjustments():
    """Load Order Size Adjustments.csv from Azure Blob Storage."""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=ORDER_SIZE_BLOB)
        csv_data = blob_client.download_blob().readall().decode("utf-8")
        df = pd.read_csv(StringIO(csv_data))
        df.columns = df.columns.str.strip()
        df["Minimum"] = pd.to_numeric(df["Minimum"], errors="coerce")
        df["Maximum"] = pd.to_numeric(df["Maximum"], errors="coerce")
        # Parse Adjustment column (e.g., "40%" -> 0.40)
        df["Adjustment_Pct"] = df["Adjustment"].astype(str).str.replace("%", "").astype(float) / 100
        return df
    except Exception as e:
        st.error(f"Error loading Order Size Adjustments: {str(e)}")
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
        if service_type == "Rewinder":
            ra_val = paper_row.get("RW_RunAdjust")
        else:
            ra_val = paper_row.get("SHT_RunAdjust")
        run_adjust = float(ra_val) if pd.notna(ra_val) else 1.0
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
        raw_rc = machine_row.get("Roll_Change_Hrs")
        roll_change_hrs = float(raw_rc) if pd.notna(raw_rc) else 0.25
        raw_su = machine_row.get("Setup_Hrs")
        setup_hrs = float(raw_su) if pd.notna(raw_su) else 0.5

        # Step 1: Calculate average roll weight
        avg_roll_weight = calculate_roll_weight(
            parent_roll_diameter,
            parent_roll_core,
            parent_roll_width,
            density_factor
        )

        # Use 20,000 lb minimum for rate calculation so the base rate
        # stays flat; order-qty bracket upcharges handle smaller orders.
        rate_qty = max(quantity_lbs, 20000)

        # Step 2: Calculate number of rolls (round up - no partial rolls)
        num_rolls = int(np.ceil(rate_qty / avg_roll_weight)) if avg_roll_weight > 0 else 1
        num_rolls = max(1, num_rolls)  # At least 1 roll

        # Step 2b: Determine rolls running at one time (for sheeting)
        # If sheeting and caliper <= 11 (or blank/zero), use NumShtrRolls
        # Otherwise, 1 roll at a time
        if equip_type == "Sheeter" and (caliper == 0 or caliper is None or caliper <= 0.011):
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
        processing_hours = rate_qty / lbs_per_hour if lbs_per_hour > 0 else 0

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

        # Step 8: $/CWT (always based on rate_qty so base rate is flat)
        base_rate_cwt = (total_cost / rate_qty) * 100 if rate_qty > 0 else 0

        # Calculate actual order values (for display)
        actual_num_rolls = int(np.ceil(quantity_lbs / avg_roll_weight)) if avg_roll_weight > 0 else 1
        actual_num_rolls = max(1, actual_num_rolls)
        actual_processing_hours = quantity_lbs / lbs_per_hour if lbs_per_hour > 0 else 0
        if equip_type == "Rewinder":
            actual_roll_changes = actual_num_rolls
        else:
            actual_roll_changes = int(np.ceil(actual_num_rolls / rolls_running)) if rolls_running > 0 else actual_num_rolls
        actual_roll_change_hours = roll_change_hrs * actual_roll_changes
        actual_total_hours = actual_processing_hours + actual_roll_change_hours + setup_hrs
        actual_total_cost = actual_total_hours * hourly_rate

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
            "num_rolls": actual_num_rolls,
            "rolls_running": rolls_running,
            "num_roll_changes": actual_roll_changes,
            "lbs_per_hour": round(lbs_per_hour, 2),
            "processing_hours": round(actual_processing_hours, 4),
            "roll_change_hours": round(actual_roll_change_hours, 4),
            "total_hours": round(actual_total_hours, 4),
            "total_cost": round(actual_total_cost, 2),
            "quantity_lbs": quantity_lbs,
            "service_type": service_type,
            "equip_type": equip_type
        }

        return result

    except Exception as e:
        result["error"] = str(e)
        return result


def calculate_additional_charges(checked_items, add_charge_df, quantity_lbs):
    """
    Calculate surcharges for checked add-charge items.

    Returns (total_additional_cwt, breakdown) where breakdown is a list of
    (parameter_name, charge_cwt) tuples.
    """
    breakdown = []
    total = 0.0

    for param_name in checked_items:
        row = add_charge_df[add_charge_df["Parameter"].str.strip() == param_name]
        if row.empty:
            continue
        row = row.iloc[0]
        method = str(row.get("Charge Method", "")).strip()
        charge_val = row.get("Additional Charge", 0)

        if method == "CWT":
            amt = clean_currency(charge_val, 0)
        elif method == "Formula":
            # Partial Roll Usage formula: (60 / Order Qty) * 100
            if quantity_lbs > 0:
                amt = (60 / quantity_lbs) * 100
            else:
                amt = 0.0
        else:
            amt = 0.0

        breakdown.append((param_name, round(amt, 2)))
        total += amt

    return round(total, 2), breakdown


def calculate_auto_charges(base_rate_cwt, service_type, quantity_lbs,
                           parent_roll_width, cut_width, sheet_length,
                           hourly_rate, order_size_df):
    """
    Apply order-qty bracket multipliers, 1-hour minimum, and auto-calculated surcharges.

    Returns (adjusted_base_rate, auto_charges_cwt, breakdown_list).
    """
    breakdown = []
    auto_charges = 0.0

    # --- Order Qty bracket upcharge (from Order Size Adjustments table) ---
    adjusted_base = base_rate_cwt  # base rate stays unchanged
    if order_size_df is not None:
        machine_group = "Rewinding" if service_type == "Rewinder" else "Sheeting"
        qty_rows = order_size_df[
            (order_size_df["MachineGroup"].str.strip() == machine_group)
            & (order_size_df["AdjDescription"].str.strip() == "OrderQty")
            & (order_size_df["Minimum"] <= quantity_lbs)
            & (order_size_df["Maximum"] >= quantity_lbs)
        ]
        if not qty_rows.empty:
            adj_pct = qty_rows.iloc[0]["Adjustment_Pct"]
            if adj_pct > 0:
                upcharge_amt = round(base_rate_cwt * adj_pct, 2)
                auto_charges += upcharge_amt
                breakdown.append((f"Order Qty Adjustment (+{int(adj_pct * 100)}%)", upcharge_amt))

    # --- 1-hour minimum charge ---
    # After order qty upcharge, check if (base + upcharge) × qty covers 1 hour.
    # If not, replace all prior charges with a single minimum line.
    subtotal_cwt = adjusted_base + auto_charges
    order_total = (subtotal_cwt * quantity_lbs) / 100
    if order_total < hourly_rate and quantity_lbs > 0:
        min_rate_cwt = round((hourly_rate / quantity_lbs) * 100, 2)
        min_upcharge = round(min_rate_cwt - adjusted_base, 2)
        auto_charges = min_upcharge
        breakdown = [("Minimum order charge applied", min_upcharge)]

    # --- Trim charge (Sheeting only) ---
    if service_type != "Rewinder" and parent_roll_width > 0 and cut_width > 0:
        trim = parent_roll_width - cut_width
        if trim > 4:
            auto_charges += 2.20
            breakdown.append(("Trim > 4\" surcharge", 2.20))
        elif trim < 0.5:
            trim_charge = round(base_rate_cwt * 0.165, 2)
            auto_charges += trim_charge
            breakdown.append(("Insufficient Trim surcharge", trim_charge))

    # --- Sheet Length charge (Sheeting only) ---
    if service_type != "Rewinder" and 0 < sheet_length < 21:
        length_charge = round(base_rate_cwt * 0.20, 2)
        auto_charges += length_charge
        breakdown.append(("Short Sheet Length surcharge", length_charge))

    return round(adjusted_base, 2), round(auto_charges, 2), breakdown


# =========================================================
# MAIN APP
# =========================================================
def main():
    st.title("NCC Converting Quote")

    # Key suffix — changes on Reset to force fresh widgets
    ks = f"_{st.session_state.reset_counter}"

    # Load data
    paper_df = load_paper_info()
    machine_df = load_machine_info()
    add_charge_df = load_add_charges()
    order_size_df = load_order_size_adjustments()

    if paper_df is None or machine_df is None:
        st.error("Failed to load required data. Please check Azure connection.")
        st.stop()

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
        label_visibility="collapsed",
        key=f"service_selection{ks}"
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
            index=0 if product_groups else None,
            key=f"product_group{ks}"
        )

        # Basis Weight with unit toggle
        basis_weight_unit = st.radio(
            "Basis Weight Unit",
            options=["LBS", "GSM"],
            horizontal=True,
            key=f"basis_weight_unit{ks}"
        )
        basis_weight = st.number_input(
            f"Basis Weight ({basis_weight_unit})",
            min_value=0.0,
            value=0.0,
            step=0.1,
            format="%.2f",
            key=f"basis_weight{ks}"
        )

        caliper = st.number_input(
            "Caliper (PTS)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            format="%.2f",
            key=f"caliper{ks}"
        )

        if service_type == "Sheeter":
            sheet_width = st.number_input(
                "Sheet Width (IN)",
                min_value=0.0,
                value=0.0,
                step=0.25,
                format="%.2f",
                key=f"sheet_width{ks}"
            )
            sheet_length = st.number_input(
                "Sheet Length (IN)",
                min_value=0.0,
                value=0.0,
                step=0.25,
                format="%.2f",
                key=f"sheet_length{ks}"
            )
        else:
            cut_width = st.number_input(
                "Cut Width Total (IN)",
                min_value=0.0,
                value=0.0,
                step=0.25,
                format="%.2f",
                key=f"cut_width{ks}"
            )
            sheet_width = 0.0
            sheet_length = 0.0

    with col2:
        parent_roll_width = st.number_input(
            "Parent Roll Width (IN)",
            min_value=0.0,
            value=0.0,
            step=0.25,
            format="%.2f",
            key=f"parent_roll_width{ks}"
        )

        parent_roll_diameter = st.number_input(
            "Parent Roll Diameter (IN)",
            min_value=0.0,
            value=0.0,
            step=0.5,
            format="%.2f",
            key=f"parent_roll_diameter{ks}"
        )

        parent_roll_core = st.number_input(
            "Parent Roll Core (IN)",
            min_value=0.0,
            value=0.0,
            step=0.5,
            format="%.2f",
            key=f"parent_roll_core{ks}"
        )

        quantity_lbs = st.number_input(
            "Quantity - Raw Material (LBS)",
            min_value=0.0,
            value=0.0,
            step=100.0,
            format="%.0f",
            key=f"quantity_lbs{ks}"
        )

        # Additional Charges checkboxes (in right column)
        checked_items = []
        if add_charge_df is not None:
            operation_pattern = "Rewind" if service_type == "Rewinder" else "Sheeting"
            checkbox_rows = add_charge_df[
                (add_charge_df["Input Method"].str.strip() == "Checkbox")
                & (add_charge_df["Operation"].str.strip().str.contains(operation_pattern, case=False, na=False))
            ]
            if not checkbox_rows.empty:
                st.markdown("**Additional Charges**")
                for _, row in checkbox_rows.iterrows():
                    param = str(row["Parameter"]).strip()
                    if st.checkbox(param, key=f"chk_{param}{ks}"):
                        checked_items.append(param)
        if not checked_items:
            checked_items = []

    # For sheeting, calculate num_out and cut_width_total from sheet_width
    if service_type == "Sheeter" and sheet_width > 0 and parent_roll_width > 0:
        num_out = int(parent_roll_width // sheet_width)
        cut_width = sheet_width * num_out
        trim = parent_roll_width - cut_width
        st.info(f"**{num_out} out** — Cut Width Total: {cut_width:.2f}\" | Trim: {trim:.2f}\"")
    elif service_type == "Sheeter":
        num_out = 0
        cut_width = 0.0

    st.divider()

    # =========================================================
    # CALCULATE QUOTE / RESET
    # =========================================================
    btn_col1, btn_col2 = st.columns([3, 1])
    with btn_col1:
        calculate_clicked = st.button("Calculate Quote", type="primary", use_container_width=True)
    with btn_col2:
        if st.button("Reset", use_container_width=True):
            st.session_state.reset_counter += 1
            st.session_state.quote_result = None
            st.session_state.quote_params = {}
            st.rerun()

    if calculate_clicked:
        # Validate inputs
        errors = []
        if basis_weight <= 0:
            errors.append("Basis Weight must be greater than 0")
        if service_type == "Sheeter" and sheet_width <= 0:
            errors.append("Sheet Width must be greater than 0")
        elif service_type != "Sheeter" and cut_width <= 0:
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
                "sheet_width": sheet_width,
                "sheet_length": sheet_length,
                "parent_roll_width": parent_roll_width,
                "parent_roll_diameter": parent_roll_diameter,
                "parent_roll_core": parent_roll_core if parent_roll_core > 0 else 3.0,
                "quantity_lbs": quantity_lbs,
                "service_type": service_type
            }

            # Calculate
            result = calculate_base_rate(params, paper_df, machine_df, product_group_col)

            if result["success"]:
                # Auto charges (order qty brackets, trim, sheet length)
                adjusted_base, auto_total, auto_breakdown = calculate_auto_charges(
                    result["base_rate_cwt"], service_type, quantity_lbs,
                    parent_roll_width, cut_width, sheet_length,
                    result["details"]["hourly_rate"], order_size_df
                )
                result["adjusted_base_cwt"] = adjusted_base
                result["auto_charges_cwt"] = auto_total
                result["auto_breakdown"] = auto_breakdown

                # Checkbox surcharges
                if checked_items and add_charge_df is not None:
                    add_total, add_breakdown = calculate_additional_charges(
                        checked_items, add_charge_df, quantity_lbs
                    )
                else:
                    add_total, add_breakdown = 0.0, []
                result["additional_charges_cwt"] = add_total
                result["additional_breakdown"] = add_breakdown

                # Total = adjusted base + auto surcharges + checkbox surcharges
                result["total_rate_cwt"] = round(adjusted_base + auto_total + add_total, 2)
            else:
                result["adjusted_base_cwt"] = 0.0
                result["auto_charges_cwt"] = 0.0
                result["auto_breakdown"] = []
                result["additional_charges_cwt"] = 0.0
                result["additional_breakdown"] = []
                result["total_rate_cwt"] = 0.0

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
            total_rate = result.get("total_rate_cwt", result["base_rate_cwt"])
            adjusted_base = result.get("adjusted_base_cwt", result["base_rate_cwt"])
            auto_total = result.get("auto_charges_cwt", 0)
            add_total = result.get("additional_charges_cwt", 0)
            has_extras = (adjusted_base != result["base_rate_cwt"]) or auto_total > 0 or add_total > 0

            # Compute sheeting metrics for display
            details = result.get("details", {})
            mweight = None
            price_per_m = None
            if details.get("equip_type") == "Sheeter":
                params = st.session_state.quote_params
                s_width = params.get("sheet_width", 0)
                s_length = params.get("sheet_length", 0)
                area_in = details.get("area_in", 0)
                bw_lbs = details.get("basis_weight_lbs", 0)
                if s_width > 0 and s_length > 0 and area_in > 0 and bw_lbs > 0:
                    mweight = round(((s_width * s_length) / area_in) * bw_lbs * 2)
                    price_per_m = total_rate * 0.01 * mweight

            # Display metrics row
            if mweight is not None:
                col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            else:
                col_r1, col_r2 = st.columns(2)

            with col_r1:
                st.metric(label="Base Rate", value=f"${result['base_rate_cwt']:.2f} / CWT")
            with col_r2:
                if has_extras:
                    st.metric(label="Total Rate", value=f"${total_rate:.2f} / CWT")
                else:
                    st.metric(label="Total Rate", value=f"${result['base_rate_cwt']:.2f} / CWT")

            if mweight is not None:
                with col_r3:
                    st.metric(label="MWT", value=f"{mweight:,} lbs")
                with col_r4:
                    st.metric(label="Price / M Sheets", value=f"${price_per_m:,.2f}")

            # Rate breakdown
            if has_extras:
                st.markdown("**Rate Breakdown:**")
                st.write(f"- Base Rate: ${result['base_rate_cwt']:.2f}/CWT")

                # Auto charges (order qty, trim, sheet length)
                auto_breakdown = result.get("auto_breakdown", [])
                for name, amt in auto_breakdown:
                    st.write(f"- {name}: +${amt:.2f}/CWT")

                # Checkbox surcharges
                add_breakdown = result.get("additional_breakdown", [])
                for name, amt in add_breakdown:
                    st.write(f"- {name}: +${amt:.2f}/CWT")

                st.write(f"- **Total Rate: ${total_rate:.2f}/CWT**")

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
