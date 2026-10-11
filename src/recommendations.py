"""Rule-based, evidence-linked segment playbooks and individual experience guidance.

Principles
* Every statement is generated from observed statistics (segment vs. all buyers).
* Observed behaviour and ML inference are kept apart: segment statements describe the
  *group*; they are never presented as facts about one person.
* Nothing here infers income, wealth, affordability or creditworthiness.
* Every recommendation carries a customer-benefit rationale and a contact-pressure guardrail.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

THRESH_SD = 0.25  # |segment mean - overall mean| in overall-SD units to call a dimension High/Low


def level(profile: dict[str, Any], overall: dict[str, Any], var: str) -> str:
    """High / Typical / Low relative to all buyers (standardised difference)."""
    sd = overall["sd"].get(var, 0) or 1e-9
    z = (profile["means"][var] - overall[var]) / sd
    return "High" if z >= THRESH_SD else "Low" if z <= -THRESH_SD else "Typical"


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _usd(x: float) -> str:
    return f"${x:,.0f}"


def segment_levels(p: dict[str, Any], o: dict[str, Any]) -> dict[str, str]:
    """Transparent customer-intelligence profile: each dimension is a *relative* band."""
    intl = p["means"]["is_international"]
    return {
        "Investment orientation": level(p, o, "is_investment_purpose"),
        "Property activity": level(p, o, "purchase_count"),
        "Purchase ticket size": level(p, o, "average_purchase_price_usd"),
        "Commercial (office) preference": level(p, o, "office_share"),
        "Financing dependency": level(p, o, "is_loan_applied"),
        "Customer satisfaction": level(p, o, "satisfaction_score"),
        "Purchase time-span": level(p, o, "purchase_span_months"),
        "Geographic concentration": "International mix" if intl >= o["is_international"] + 0.08 else
                                    "Mostly domestic (USA)" if intl <= o["is_international"] - 0.08 or intl < 0.2 else "Mixed",
    }


def segment_insights(p: dict[str, Any], o: dict[str, Any]) -> list[str]:
    """Plain-language findings, each tied to numbers. Only differences that are material are stated."""
    m, ins = p["means"], []
    lv = segment_levels(p, o)
    if lv["Property activity"] != "Typical":
        ins.append(f"Purchase activity is {lv['Property activity'].lower()}er than other buyers: "
                   f"{m['purchase_count']:.1f} purchases on average vs {o['purchase_count']:.1f} overall.")
    if lv["Purchase ticket size"] != "Typical":
        ins.append(f"Average purchase value is {lv['Purchase ticket size'].lower()}er: {_usd(m['average_purchase_price_usd'])} "
                   f"vs {_usd(o['average_purchase_price_usd'])} overall (observed transactions only).")
    if lv["Commercial (office) preference"] != "Typical":
        ins.append(f"Office purchases account for {_pct(m['office_share'])} of units vs {_pct(o['office_share'])} overall.")
    if lv["Purchase time-span"] != "Typical":
        ins.append(f"Purchases are spread over {m['purchase_span_months']:.0f} months on average vs {o['purchase_span_months']:.0f} overall.")
    ins.append(f"{_pct(m['is_loan_applied'])} of this segment applied for a loan vs {_pct(o['is_loan_applied'])} overall"
               + (" — a material difference." if lv["Financing dependency"] != "Typical" else " — no material difference."))
    ins.append(f"{_pct(m['is_investment_purpose'])} acquired for investment vs {_pct(o['is_investment_purpose'])} overall"
               + (" — a material difference." if lv["Investment orientation"] != "Typical" else " — no material difference."))
    ins.append(f"Average satisfaction is {m['satisfaction_score']:.2f}/5 vs {o['satisfaction_score']:.2f} overall.")
    if p["same_tower_repeat_share"] > 0.2:
        ins.append(f"{_pct(p['same_tower_repeat_share'])} bought more than one unit in the same tower.")
    return ins


def segment_playbook(p: dict[str, Any], o: dict[str, Any]) -> dict[str, Any]:
    """Needs, friction, communication, property, financing, engagement and marketing rules."""
    m, lv = p["means"], segment_levels(p, o)
    needs, friction, comm, prop, fin, eng, mkt = [], [], [], [], [], [], []

    # --- activity / time-span
    if lv["Property activity"] == "High":
        needs.append("A consolidated view of several holdings and a smoother repeat-purchase process.")
        friction.append("Repeating the same paperwork and onboarding for each additional purchase.")
        eng.append("Offer a single point of contact and pre-filled details for follow-on purchases.")
    elif lv["Property activity"] == "Low":
        needs.append("Clear guidance for a smaller number of purchase decisions.")
        eng.append("Keep communication focused on the specific listings already viewed.")
    if lv["Purchase time-span"] == "High":
        comm.append("Milestone-based updates (new availability matching past purchases) rather than frequent reminders.")
    elif lv["Purchase time-span"] == "Low":
        comm.append("Timely availability alerts: this segment completes its purchases in a shorter window.")

    # --- price / size
    avg_p = _usd(m["average_purchase_price_usd"])
    if lv["Purchase ticket size"] == "High":
        prop.append(f"Surface units in the upper part of the observed price range (segment average {avg_p}).")
        needs.append("Detailed unit information (area, price per sq ft, comparable units) to support larger decisions.")
        friction.append("Thin or generic listings when each transaction is of higher value.")
    elif lv["Purchase ticket size"] == "Low":
        prop.append(f"Surface units near the segment's typical price point (average {avg_p}); avoid showing out-of-pattern listings.")
        needs.append("Easy price comparison between similar units.")
        friction.append("Being shown listings far above the price range they have bought in.")
    else:
        prop.append(f"Match recommendations to the observed price point (segment average {avg_p}).")
    prop.append(f"Typical unit size is about {m['average_area_sqft']:.0f} sq ft at {_usd(m['average_price_per_sqft_usd'])} per sq ft.")

    # --- commercial
    if lv["Commercial (office) preference"] == "High":
        prop.append("Include office units in recommendations; office purchases are more common here than overall.")
        mkt.append("Commercial-property campaigns with floor-area and tower information.")
    elif lv["Commercial (office) preference"] == "Low":
        prop.append("Prioritise apartments; office units are less common in this segment's purchases.")
    else:
        prop.append("Offer both apartments and a limited number of office units, matching past category mix.")

    # --- investment orientation
    if lv["Investment orientation"] == "High":
        mkt.append("Investment-focused property content (portfolio view, unit comparison).")
        needs.append("Comparable-unit data and transparent pricing information.")
    elif lv["Investment orientation"] == "Low":
        mkt.append("Home-oriented content (living space, tower facilities, neighbourhood).")
    else:
        mkt.append("Mixed content: the investment/home split matches the overall population, so let each client's own stated purpose steer the message.")

    # --- financing
    if lv["Financing dependency"] == "High":
        fin.append(f"Financing is more common here ({_pct(m['is_loan_applied'])} vs {_pct(o['is_loan_applied'])}): lead with financing information and clear process timelines.")
        friction.append("Unclear financing steps or late disclosure of financing requirements.")
    elif lv["Financing dependency"] == "Low":
        fin.append(f"Financing is less common ({_pct(m['is_loan_applied'])} vs {_pct(o['is_loan_applied'])}): keep financing as an optional, low-prominence message.")
    else:
        fin.append(f"Loan usage ({_pct(m['is_loan_applied'])}) does not differ materially from the overall rate ({_pct(o['is_loan_applied'])}): "
                   "decide financing messaging per client from the client's own loan flag, not from the segment.")

    # --- satisfaction
    if lv["Customer satisfaction"] == "Low":
        eng.append("Prioritise service follow-up and feedback collection before further promotion.")
        friction.append("Below-average satisfaction suggests unresolved service issues.")
    elif lv["Customer satisfaction"] == "High":
        eng.append("Invite feedback and optional referrals; avoid over-contacting a satisfied group.")
    else:
        eng.append("Satisfaction is in line with overall; monitor low individual scores for service recovery.")
    sat_low = sum(v for k, v in p["satisfaction_distribution"].items() if int(k) <= 2)
    eng.append(f"{_pct(sat_low)} of the segment rated satisfaction 1-2.")

    # --- geography / channel
    if lv["Geographic concentration"] == "International mix":
        comm.append("Use time-zone-friendly, language-clear communication and remote-friendly purchase steps.")
        friction.append("Process steps that assume local presence.")
    top_ref = max(p["referral_share"], key=p["referral_share"].get)
    comm.append(f"Largest acquisition channel is '{top_ref}' ({_pct(p['referral_share'][top_ref])}); "
                "use the channel each client prefers where consent exists.")
    if m["is_company"] > o["is_company"] + 0.05:
        eng.append("Assign account-style service for corporate buyers.")

    return {
        "profile_levels": lv,
        "insights": segment_insights(p, o),
        "customer_needs": needs or ["No segment-specific need is supported by the data; use individual signals."],
        "potential_friction": friction or ["No segment-specific friction is supported by the data."],
        "communication_strategy": comm, "property_strategy": prop, "financing_strategy": fin,
        "engagement_strategy": eng, "marketing_recommendation": mkt,
        "customer_value_guardrails": [
            "Cap promotional contact frequency; send only items matching the client's observed pattern.",
            "Describe recommendations as 'similar to units others in your group bought', never as assumptions about means or ability to pay.",
            "Respect opt-outs and explain why a recommendation is shown.",
        ],
    }


def client_experience(row: pd.Series, playbook: dict[str, Any]) -> dict[str, list[str]]:
    """Individual layer — uses only the client's OWN recorded fields, kept separate from segment inference."""
    obs, actions = [], []
    obs.append(f"Acquisition purpose: {row['acquisition_purpose']}; loan applied: {row['loan_applied']}; "
               f"satisfaction: {int(row['satisfaction_score'])}/5; referral: {row['referral_channel']}.")
    if row["loan_applied"] == "Yes":
        actions.append("Provide clear financing/process information (client applied for a loan).")
    if row["acquisition_purpose"] == "Investment":
        actions.append("Include comparison data and portfolio-style content (client stated investment purpose).")
    else:
        actions.append("Keep content home-oriented (client stated personal use).")
    if int(row["satisfaction_score"]) <= 2:
        actions.append("Service-recovery follow-up before any promotion (low recorded satisfaction).")
    elif int(row["satisfaction_score"]) == 5:
        actions.append("Optional feedback/referral invitation (top satisfaction rating).")
    if row["client_type"] == "Company":
        actions.append("Use account-style communication (corporate client).")
    return {"observed": obs, "actions": actions}


def describe_segment(p: dict[str, Any], o: dict[str, Any]) -> tuple[str, dict[str, str]]:
    """Short description and the 'key characteristics' block, generated from the segment's statistics."""
    m, lv = p["means"], segment_levels(p, o)
    top = [t["label"].lower() + (" higher" if t["effect"] > 0 else " lower") for t in p["distinguishing_traits"][:3]]
    desc = (f"{p['n']:,} buyers ({_pct(p['share'])} of all). Compared with other buyers: "
            + (", ".join(top) if top else "no strongly distinguishing trait") + ". "
            f"Loan use ({_pct(m['is_loan_applied'])}) and investment purpose ({_pct(m['is_investment_purpose'])}) "
            "are close to the overall rates, so these are best handled per client.")
    chars = {
        "Average age": f"{m['age']:.1f}", "Purchase count": f"{m['purchase_count']:.2f}",
        "Average purchase value": _usd(m["average_purchase_price_usd"]), "Investment rate": _pct(m["is_investment_purpose"]),
        "Loan rate": _pct(m["is_loan_applied"]), "Office share": _pct(m["office_share"]),
        "Satisfaction": f"{m['satisfaction_score']:.2f} / 5", "Top region": next(iter(p["top_regions"])),
    }
    return desc, chars


# --------------------------------------------------------------------------------------------
# Structured, auditable recommendations (one object per recommendation).
# --------------------------------------------------------------------------------------------
EXCLUDED_INPUTS = ("gender", "country", "region", "age", "nationality", "inferred wealth or income")
MIN_ASSIGNMENT_STABILITY = 0.80   # below this expected stability the segment is treated as ambiguous
REPEAT_BUYER_MIN_PURCHASES = 5    # about the top 6% of buyers in this dataset


def _rec(rid: str, basis: str, action: str, evidence: list[str], customer_benefit: str, business_benefit: str,
         confidence: str, rule: str, limitations: str, human_review: str | None = None) -> dict[str, Any]:
    return {"id": rid, "basis": basis, "action": action, "evidence": evidence, "customer_benefit": customer_benefit,
            "business_benefit": business_benefit, "confidence": confidence, "rule": rule,
            "limitations": limitations, "human_review": human_review or "Not required; low-risk, reversible communication choice."}


def structured_recommendations(row: pd.Series, profile: dict[str, Any], overall: dict[str, Any],
                               est_stability: float | None = None,
                               min_stability: float = MIN_ASSIGNMENT_STABILITY) -> list[dict[str, Any]]:
    """Evidence-backed actions for one buyer.

    ``basis`` separates *observed* (the buyer's own recorded fields; no model involved) from
    *ML-derived inference* (depends on the segment assignment). Nothing here uses gender,
    country/region, age or any inferred income/wealth (see ``EXCLUDED_INPUTS``).
    If the assignment is ambiguous, segment-based personalisation is withheld and a clarification
    action is returned instead.
    """
    recs: list[dict[str, Any]] = []
    own = "Recorded in the client's own record."
    lim_obs = "Reflects only what was recorded; the client's circumstances may have changed."

    sat = int(row["satisfaction_score"])
    if sat <= 2:
        recs.append(_rec("OBS-LOW-SAT", "observed", "Offer a service-recovery follow-up before any promotional contact.",
                         [f"Recorded satisfaction is {sat}/5. {own}"],
                         "Concerns are heard before being marketed to.", "Reduces churn/complaint risk; protects trust.",
                         "High: recorded fact", "Rule: satisfaction <= 2", lim_obs + " The score has no stated reason attached.",
                         "Recommended: a person should review the account and the reason for the score."))
    elif sat == 5:
        recs.append(_rec("OBS-HIGH-SAT", "observed", "Offer an optional feedback or referral invitation.",
                         [f"Recorded satisfaction is 5/5. {own}"], "A low-effort way to share a good experience.",
                         "Possible referral source.", "High: recorded fact", "Rule: satisfaction == 5",
                         lim_obs + " Do not over-contact; the invitation must remain optional."))
    if row["loan_applied"] == "Yes":
        recs.append(_rec("OBS-LOAN", "observed", "Provide clear financing steps, documents and timelines.",
                         [f"Client recorded a loan application. {own}"], "Fewer surprises in the financing process.",
                         "Fewer delays and drop-offs in financed transactions.", "High: recorded fact",
                         "Rule: loan_applied == Yes", lim_obs + " Does not imply anything about ability to pay."))
    if row["acquisition_purpose"] == "Investment":
        recs.append(_rec("OBS-INVEST", "observed", "Include unit-comparison data and a portfolio-style summary.",
                         [f"Stated acquisition purpose: Investment. {own}"], "Information suited to comparing holdings.",
                         "Higher relevance of investment-oriented content.", "High: recorded fact",
                         "Rule: acquisition_purpose == Investment", lim_obs))
    if int(row["purchase_count"]) >= REPEAT_BUYER_MIN_PURCHASES:
        recs.append(_rec("OBS-REPEAT", "observed", "Offer a consolidated portfolio summary and a single point of contact.",
                         [f"{int(row['purchase_count'])} recorded purchases (threshold {REPEAT_BUYER_MIN_PURCHASES}+, "
                          f"about the top 6% of buyers); overall mean is {overall['purchase_count']:.1f}."],
                         "Less repeated paperwork and a clear view of holdings.", "Smoother repeat-purchase process.",
                         "High: recorded fact", f"Rule: purchase_count >= {REPEAT_BUYER_MIN_PURCHASES}",
                         "Threshold is a convenience cut-off, not a validated tier."))
    if row["client_type"] == "Company":
        recs.append(_rec("OBS-COMPANY", "observed", "Use account-style communication with a named contact.",
                         [f"Client type is Company. {own}"], "Service matched to organisational buying.",
                         "Cleaner handling of corporate accounts.", "High: recorded fact", "Rule: client_type == Company", lim_obs))

    if sat <= 2:  # service recovery first: defer promotional / personalisation content
        recs = [r for r in recs if r["id"] != "OBS-INVEST"]
        recs[0]["limitations"] += " Promotional and segment-based content is deliberately withheld until the issue is reviewed."
        return recs

    stable = est_stability is None or np.isnan(est_stability) or est_stability >= min_stability
    if not stable:
        recs.append(_rec("AMBIG-ASSIGN", "ML-derived inference",
                         "Ask the client about preferences instead of personalising from the segment.",
                         [f"Expected segment-assignment stability is {est_stability:.0%} (< {min_stability:.0%}): the buyer sits near a "
                          "boundary between segments."], "Avoids being shown content based on a shaky grouping.",
                         "Avoids wasted or irrelevant personalisation.", f"Medium: stability estimate {est_stability:.0%}",
                         "Rule: est_assignment_stability < threshold (src/predict.py calibration table)",
                         "Segments are a description of observed behaviour, not a verified preference.",
                         "Recommended before any segment-based targeting."))
        return recs

    seg_conf = ("Estimated assignment stability " + (f"{est_stability:.0%}" if est_stability is not None and not np.isnan(est_stability) else "n/a")
                + "; effect sizes below are descriptive, not causal")
    lim_seg = "Segment behaviour is an average; this buyer may differ. Segment = observed purchases, not stated preferences."
    m, o, lv = profile["means"], overall, segment_levels(profile, overall)
    if lv["Purchase ticket size"] != "Typical":
        hi = lv["Purchase ticket size"] == "High"
        recs.append(_rec("SEG-TICKET", "ML-derived inference",
                         "Show units near the segment's typical price point" + (" (upper end of the observed range)." if hi else "; avoid listings far above it."),
                         [f"Segment average purchase ${m['average_purchase_price_usd']:,.0f} vs ${o['average_purchase_price_usd']:,.0f} overall.",
                          f"Segment: {profile['name']} ({profile['n']:,} buyers)."], "Listings that resemble what similar buyers purchased.",
                         "More relevant property matching.", seg_conf, "Rule: |segment mean - overall mean| > 0.25 SD (ticket size)", lim_seg))
    if lv["Commercial (office) preference"] != "Typical":
        hi = lv["Commercial (office) preference"] == "High"
        recs.append(_rec("SEG-CATEGORY", "ML-derived inference",
                         "Include office units alongside apartments." if hi else "Prioritise apartments in recommendations.",
                         [f"Office share of purchases {m['office_share']:.0%} vs {o['office_share']:.0%} overall."],
                         "Category mix that matches similar buyers' purchases.", "More relevant property matching.", seg_conf,
                         "Rule: |segment mean - overall mean| > 0.25 SD (office share)", lim_seg))
    if lv["Purchase time-span"] != "Typical":
        hi = lv["Purchase time-span"] == "High"
        recs.append(_rec("SEG-CADENCE", "ML-derived inference",
                         "Use milestone-based updates rather than frequent reminders." if hi else "Send timely availability alerts; purchases cluster in a short window.",
                         [f"Purchases span {m['purchase_span_months']:.0f} months on average vs {o['purchase_span_months']:.0f} overall."],
                         "Contact frequency that matches how the segment buys.", "Avoids over-contacting; better timing.", seg_conf,
                         "Rule: |segment mean - overall mean| > 0.25 SD (purchase span)", lim_seg))
    if not any(r["basis"] == "ML-derived inference" for r in recs):
        recs.append(_rec("SEG-NONE", "ML-derived inference", "No segment-specific action is supported; rely on the client's own recorded fields.",
                         [f"{profile['name']} does not differ from the overall population by more than 0.25 SD on any profile dimension."],
                         "Avoids generic personalisation presented as insight.", "Avoids unsupported targeting.", seg_conf,
                         "Rule: no dimension above the 0.25 SD threshold", lim_seg))
    return recs
