"""Deterministic synthetic ophthalmic notes with gold PHI spans and protected clinical values.

Every value comes from the built-in fictional lists below; the generator reads no input files.
Gold labeling policy (conservative, i2b2-style): every person name (patient, clinician and
staff), every date, ages over 89, record numbers, phone numbers, email addresses, street
addresses (including city, state and ZIP) and device serial numbers are PHI. Credentials,
organization names, device model names and clinical values are not.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from .config import write_jsonl
from .phi_grader import CATEGORIES, occurrences

NOTE_TYPES = ("exam_note", "imaging_report", "referral_letter")
DEFAULT_SEED = 42
DEFAULT_SIZE = 30

EPONYMS = (
    "Goldmann",
    "Humphrey 24-2",
    "Amsler grid",
    "Fuchs dystrophy",
    "Descemet membrane endothelial keratoplasty",
    "Stargardt disease",
    "Krukenberg spindle",
    "Graves orbitopathy",
    "Hollenhorst plaque",
    "Sjogren syndrome",
    "Schirmer test",
)
ACUITY_RE = re.compile(r"^20/\d{2,3}$")

# --- Fictional value lists -------------------------------------------------------------

_FEMALE_FIRST = (
    "Margaret", "Priya", "Lucinda", "Yuki", "Bernadette", "Ingrid", "Dolores", "Mei-Ling",
    "Svetlana", "Anneliese", "Rosalind", "Theodora", "Imelda", "Ngozi", "Harriet",
)
_MALE_FIRST = (
    "Desmond", "Tomasz", "Rafael", "Kwame", "Arturo", "Oluwaseun", "Cornelius", "Thaddeus",
    "Ezekiel", "Bartholomew", "Ignatius", "Leopold", "Matthias", "Hiroshi", "Alistair",
)
_PATIENT_LAST = (
    "Okafor", "Brannigan", "Venkataraman", "Wojcik", "Albright", "Quintero", "Tanabe",
    "Fairweather", "Mensah", "Halvorsen", "Castellano", "Pruitt", "Zhou", "Adeyemi", "Volkova",
    "Ashby", "Lindgren", "Merriweather", "Delacroix", "Nakashima", "Abernathy", "Szabo",
    "Okonkwo", "Rasmussen", "Villanueva",
)
_PHYSICIANS = (
    ("Henry", "Lindqvist"), ("Anita", "Raman"), ("Samuel", "Whitcombe"), ("Farah", "Haddad"),
    ("Gregory", "Osei"), ("Helena", "Marchetti"), ("Daniel", "Kowalczyk"), ("Naomi", "Feldstein"),
    ("Victor", "Agarwal"), ("Claire", "Dunmore"), ("Ruth", "Okoye"), ("Peter", "Sandoval"),
)
_OPTOMETRISTS = (
    ("Lena", "Petrakis"), ("Marcus", "Ellery"), ("Joanna", "Reyes-Whitfield"), ("Tobias", "Engstrom"),
)
_TECHNICIANS = ("K. Mbeki", "R. Tran", "S. Oyelaran", "J. Varga", "D. Moreau")
_STREETS = (
    "Birch Hollow", "Larkspur", "Old Mill", "Juniper Ridge", "Copperfield", "Harborview",
    "Sycamore Bend", "Quarry Hill", "Meadowlark", "Fernwood",
)
_STREET_SUFFIXES = ("Lane", "Road", "Drive", "Avenue", "Court", "Way", "Street")
_CITIES = (
    ("Alder Creek", "OR"), ("Brookhaven", "WA"), ("Cedar Hollow", "ID"), ("Larch Valley", "MN"),
    ("Millbrook", "OH"), ("Westhaven", "PA"),
)
_AREA_CODES = ("503", "971", "360", "541", "612", "216")
_REFERRING_PRACTICES = (
    ("Cascade Family Eye Care", "cascadefamilyeye"),
    ("Brightview Optometry", "brightviewoptometry"),
    ("Clearwater Vision Clinic", "clearwatervision"),
)
_SPECIALTY_PRACTICES = {
    "retina": "Lakeside Retina Associates",
    "glaucoma": "Summit Glaucoma Center",
    "cornea": "Riverbend Cornea Specialists",
    "oculoplastics": "Northgate Oculoplastics",
}
_DEVICES = (("SD-OCT Model A", "SDA"), ("SS-OCT Model B", "SSB"))
_ACUITY = ("20/20", "20/25", "20/30", "20/40", "20/50", "20/60", "20/80", "20/100", "20/200", "20/400")
_GOOD_ACUITY = _ACUITY[:4]
_TONOMETRY = ("Goldmann", "Goldmann", "Goldmann", "iCare", "Tono-Pen")
_INTERVALS = ("4 weeks", "6 weeks", "3 months", "6 months")
_MONTHS = (
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December",
)

# Clinical scenarios. Format fields are filled with laterality, medication and eponym markers.
_SCENARIOS = (
    {
        "ages": (55, 97),
        "cc": "Glaucoma follow-up. No new complaints.",
        "hpi": "primary open-angle glaucoma {OU} on {m0} and {m1}. Reports good adherence; denies eye pain or halos.",
        "meds": (("latanoprost", "0.005% 1 gtt {OU} qhs"), ("timolol", "0.5% 1 gtt {OU} bid")),
        "eponyms": ("Humphrey 24-2",),
        "sle": "{OU}: quiet conjunctiva, deep and quiet AC, 1+ NS",
        "dfe": "C/D 0.75 {eye}, 0.60 {fellow}; inferior rim thinning {eye}. Macula and periphery flat {OU}",
        "assessment": "Primary open-angle glaucoma, moderate stage {OU}. IOP at target. Continue {m0} and {m1}; repeat {epo} and RNFL OCT in 6 months",
        "scan": "3D Disc 6.0 x 6.0 mm (RNFL)",
        "find_eye": "Average RNFL {r_thin} microns with inferior thinning.",
        "find_fellow": "Average RNFL {r_ok} microns, within normal limits.",
        "impression": "Glaucomatous inferior RNFL thinning {eye}",
        "change": "stable",
        "reason": "primary open-angle glaucoma {OU} with inferior RNFL thinning {eye}",
        "request": "glaucoma management and possible SLT {eye}",
        "specialty": "glaucoma",
        "acuity": _GOOD_ACUITY,
    },
    {
        "ages": (65, 97),
        "cc": "Blurred, wavy vision {eye} for 2 weeks.",
        "hpi": "neovascular AMD {eye} on {m0} injections, last treated 7 weeks ago. Reports new distortion on home {epo} testing.",
        "meds": (("aflibercept", "2 mg intravitreal {eye} q7-8 wk"), ("AREDS2", "1 tab PO bid")),
        "eponyms": ("Amsler grid",),
        "sle": "{OU}: PCIOL, clear cornea, quiet AC",
        "dfe": "{eye}: subretinal fluid and heme along the superior arcade. {fellow}: intermediate drusen, no heme",
        "assessment": "Neovascular AMD {eye} with recurrent fluid. {m0} injection {eye} today; shorten interval to 5 weeks",
        "scan": "3D Macula 7.0 x 7.0 mm",
        "find_eye": "CST {cst_wet} microns. Subretinal fluid and PED, increased from prior.",
        "find_fellow": "CST {cst_a} microns. Drusen without fluid.",
        "impression": "Active choroidal neovascularization {eye}",
        "change": "interval worsening",
        "reason": "recurrent subretinal fluid from neovascular AMD {eye}",
        "request": "ongoing anti-VEGF management {eye}",
        "specialty": "retina",
        "acuity": ("20/60", "20/80", "20/100", "20/200"),
    },
    {
        "ages": (45, 89),
        "cc": "Diabetic eye exam; vision fluctuates.",
        "hpi": "type 2 diabetes (A1c 8.4%) with moderate NPDR {OU}. Takes {m1}.",
        "meds": (("bevacizumab", "1.25 mg intravitreal {eye}"), ("metformin", "1000 mg PO bid")),
        "eponyms": (),
        "sle": "{OU}: clear cornea, 1+ NS",
        "dfe": "Dot-blot hemorrhages and hard exudates {OU}, worse {eye}; no NVD or NVE",
        "assessment": "Moderate NPDR {OU} with center-involving DME {eye}. {m0} injection {eye} today; glucose control discussed",
        "scan": "3D Macula 6.0 x 6.0 mm",
        "find_eye": "CST {cst_wet} microns with intraretinal cysts.",
        "find_fellow": "CST {cst_a} microns. Scattered hard exudates, no edema.",
        "impression": "Center-involving diabetic macular edema {eye}",
        "change": "interval worsening",
        "reason": "center-involving diabetic macular edema {eye}",
        "request": "anti-VEGF treatment {eye}",
        "specialty": "retina",
        "acuity": ("20/40", "20/50", "20/60", "20/80"),
    },
    {
        "ages": (55, 97),
        "cc": "Morning blur {OU} that clears by afternoon.",
        "hpi": "{epo} with worsening glare and cataract {OU}. Uses {m0} nightly.",
        "meds": (("sodium chloride 5%", "ointment {OU} qhs"),),
        "eponyms": ("Fuchs dystrophy", "Descemet membrane endothelial keratoplasty"),
        "sle": "Confluent central guttae with mild stromal edema {OU}; 2+ NS {OU}",
        "dfe": "Hazy view; macula flat {OU}",
        "assessment": "{epo} {OU} with early corneal edema, worse {eye}. Discussed {epo2} (DMEK) combined with cataract surgery {eye}",
        "scan": "Anterior segment OCT, corneal pachymetry",
        "find_eye": "Central corneal thickness {cct_thick} microns with epithelial bullae.",
        "find_fellow": "Central corneal thickness {cct_ok} microns.",
        "impression": "Corneal edema {eye} consistent with endothelial decompensation",
        "change": "interval worsening",
        "reason": "{epo} {OU} with corneal edema {eye}",
        "request": "{epo2} combined with cataract surgery {eye}",
        "specialty": "cornea",
        "acuity": ("20/40", "20/50", "20/60", "20/80"),
    },
    {
        "ages": (55, 97),
        "cc": "Glare with night driving and trouble reading signs.",
        "hpi": "visually significant nuclear sclerotic cataract {eye}. Glare testing reduces acuity by two lines.",
        "meds": (("prednisolone acetate 1%", "1 gtt {eye} qid after surgery"), ("moxifloxacin", "0.5% 1 gtt {eye} qid after surgery")),
        "eponyms": (),
        "sle": "3+ NS {eye}, 2+ NS {fellow}; clear cornea {OU}",
        "dfe": "Macula flat without drusen {OU}",
        "assessment": "Visually significant cataract {eye}. Proceed with CE/PCIOL {eye}; start {m0} and {m1} after surgery",
        "scan": "3D Macula 6.0 x 6.0 mm (pre-op screen)",
        "find_eye": "CST {cst_a} microns. Normal foveal contour; no ERM.",
        "find_fellow": "CST {cst_b} microns. Normal.",
        "impression": "No macular pathology before cataract surgery {eye}",
        "change": "stable",
        "reason": "visually significant cataract {eye}",
        "request": "cataract surgery {eye}",
        "specialty": "cornea",
        "acuity": ("20/50", "20/60", "20/80", "20/100"),
    },
    {
        "ages": (24, 55),
        "cc": "Gradual central blur {OU}; trouble reading.",
        "hpi": "{epo} confirmed by ABCA4 genetic testing. Avoids vitamin A supplements.",
        "meds": (),
        "eponyms": ("Stargardt disease",),
        "sle": "Quiet {OU}",
        "dfe": "Beaten-bronze macular atrophy with pisciform flecks {OU}",
        "assessment": "{epo} {OU}, slowly progressive. Low-vision referral; UV-blocking lenses",
        "scan": "3D Macula 6.0 x 6.0 mm",
        "find_eye": "CST {cst_thin} microns. Foveal ellipsoid zone loss.",
        "find_fellow": "CST {cst_thin2} microns. Foveal ellipsoid zone loss.",
        "impression": "Bilateral foveal atrophy consistent with {epo}",
        "change": "slow progression",
        "reason": "{epo} {OU} with declining reading vision",
        "request": "low-vision rehabilitation and clinical trial eligibility",
        "specialty": "retina",
        "acuity": ("20/100", "20/200", "20/400"),
        "fellow_acuity": ("20/100", "20/200", "20/400"),
    },
    {
        "ages": (30, 60),
        "cc": "Intermittent blur after exercise.",
        "hpi": "pigment dispersion syndrome {OU} with elevated IOP. Started {m0} at the last visit.",
        "meds": (("brimonidine", "0.1% 1 gtt {OU} tid"),),
        "eponyms": ("Krukenberg spindle",),
        "sle": "{epo} {OU}; mid-peripheral iris transillumination defects {OU}; deep AC",
        "dfe": "C/D 0.5 {OU}. Macula flat",
        "assessment": "Pigment dispersion syndrome with ocular hypertension {OU}. Continue {m0}; gonioscopy next visit",
        "scan": "3D Disc 6.0 x 6.0 mm (RNFL)",
        "find_eye": "Average RNFL {r_ok} microns, within normal limits.",
        "find_fellow": "Average RNFL {r_ok2} microns, within normal limits.",
        "impression": "Normal RNFL {OU}",
        "change": "stable",
        "reason": "pigment dispersion syndrome with ocular hypertension {OU}",
        "request": "glaucoma evaluation and consideration of SLT",
        "specialty": "glaucoma",
        "acuity": _GOOD_ACUITY,
    },
    {
        "ages": (35, 75),
        "cc": "Gritty eyes that look more prominent.",
        "hpi": "{epo} diagnosed this year; on {m0} infusions through endocrinology. Uses {m1} hourly.",
        "meds": (("teprotumumab", "IV every 3 weeks"), ("artificial tears", "1 gtt {OU} q1h prn")),
        "eponyms": ("Graves orbitopathy",),
        "sle": "Upper lid retraction and inferior punctate keratopathy {OU}",
        "dfe": "Optic nerves pink and sharp {OU}",
        "assessment": "Active {epo} {OU} without compressive optic neuropathy. Continue {m0}; lubrication",
        "scan": "3D Disc 6.0 x 6.0 mm (RNFL)",
        "find_eye": "Average RNFL {r_ok} microns, within normal limits.",
        "find_fellow": "Average RNFL {r_ok2} microns, within normal limits.",
        "impression": "No RNFL loss to suggest compressive optic neuropathy",
        "change": "stable",
        "reason": "{epo} {OU} with lid retraction and exposure keratopathy",
        "request": "oculoplastic evaluation for eyelid surgery",
        "specialty": "oculoplastics",
        "acuity": _GOOD_ACUITY,
    },
    {
        "ages": (60, 97),
        "cc": "Sudden painless loss of upper visual field {eye} this morning.",
        "hpi": "hypertension and hyperlipidemia on {m0} and {m1}. Symptoms began 6 hours before presentation.",
        "meds": (("aspirin", "81 mg PO daily"), ("atorvastatin", "40 mg PO daily")),
        "eponyms": ("Hollenhorst plaque",),
        "sle": "Quiet {OU}",
        "dfe": "{eye}: {epo} at the inferotemporal arcade with retinal whitening. {fellow}: normal",
        "assessment": "Branch retinal artery occlusion {eye}. Same-day stroke workup arranged",
        "scan": "3D Macula 7.0 x 7.0 mm",
        "find_eye": "Inner retinal hyperreflectivity and thickening inferiorly. CST {cst_a} microns.",
        "find_fellow": "Normal. CST {cst_b} microns.",
        "impression": "Acute inner retinal ischemia {eye} consistent with branch retinal artery occlusion",
        "change": "new since prior",
        "reason": "acute branch retinal artery occlusion {eye} with a visible {epo}",
        "request": "retina follow-up; a same-day stroke and carotid workup has been arranged",
        "specialty": "retina",
        "acuity": ("20/30", "20/40", "20/50", "20/60"),
    },
    {
        "ages": (40, 85),
        "cc": "Burning and foreign-body sensation {OU}.",
        "hpi": "{epo} with dry mouth; {epo2} 4 mm {OD} and 3 mm {OS}. Takes {m1} through rheumatology.",
        "meds": (("cyclosporine", "0.05% 1 gtt {OU} bid"), ("hydroxychloroquine", "200 mg PO bid")),
        "eponyms": ("Sjogren syndrome", "Schirmer test"),
        "sle": "Rapid tear breakup and inferior punctate staining {OU}",
        "dfe": "Normal {OU}; no bull's-eye maculopathy",
        "assessment": "Aqueous-deficient dry eye from {epo} {OU}. Continue {m0}; discussed punctal plugs. Annual OCT screening for {m1} toxicity",
        "scan": "3D Wide 12.0 x 9.0 mm ({m1} screening)",
        "find_eye": "Intact ellipsoid zone. CST {cst_a} microns.",
        "find_fellow": "Intact ellipsoid zone. CST {cst_b} microns.",
        "impression": "No evidence of {m1} retinopathy",
        "change": "stable",
        "reason": "{epo} with severe aqueous-deficient dry eye {OU}",
        "request": "scleral lens fitting or autologous serum tears",
        "specialty": "cornea",
        "acuity": ("20/25", "20/30", "20/40"),
    },
)

# --- Span markers ------------------------------------------------------------------------
# Templates embed PHI and protected values as inline markers; _render() strips the markers
# and records exact character offsets, so gold labels always match the text.

_MARK_RE = re.compile("\x01([A-Z0-9_]+)\x02(.*?)\x03|\x04(.*?)\x05", re.S)


def _phi(category: str, value: str) -> str:
    assert category in CATEGORIES, category
    return "\x01%s\x02%s\x03" % (category, value)


def _keep(value: str) -> str:
    return "\x04%s\x05" % value


def _render(marked: str):
    out, spans, protected, pos, length = [], [], [], 0, 0
    for match in _MARK_RE.finditer(marked):
        plain = marked[pos:match.start()]
        out.append(plain)
        length += len(plain)
        if match.group(1):
            text = match.group(2)
            spans.append({"start": length, "end": length + len(text), "category": match.group(1), "text": text})
        else:
            text = match.group(3)
            if text not in protected:
                protected.append(text)
        out.append(text)
        length += len(text)
        pos = match.end()
    out.append(marked[pos:])
    return "".join(out), spans, protected


# --- Value helpers -------------------------------------------------------------------------


def _us(d: date) -> str:
    return "%02d/%02d/%d" % (d.month, d.day, d.year)


def _us_short(d: date) -> str:
    return "%d/%d/%02d" % (d.month, d.day, d.year % 100)


def _iso(d: date) -> str:
    return d.isoformat()


def _long(d: date) -> str:
    return "%s %d, %d" % (_MONTHS[d.month - 1], d.day, d.year)


def _age_on(dob: date, on: date) -> int:
    return on.year - dob.year - ((on.month, on.day) < (dob.month, dob.day))


def _phone(rng: random.Random) -> str:
    area, line = rng.choice(_AREA_CODES), "555-01%02d" % rng.randint(0, 99)
    return rng.choice(("(%s) %s" % (area, line), "%s-%s" % (area, line)))


def _address(rng: random.Random, suite: bool) -> str:
    city, state = rng.choice(_CITIES)
    street = "%d %s %s" % (rng.randint(12, 9899), rng.choice(_STREETS), rng.choice(_STREET_SUFFIXES))
    if suite:
        street += ", Suite %d" % rng.randint(100, 450)
    return "%s, %s, %s %05d" % (street, city, state, rng.randint(10000, 99999))


@dataclass
class _Ctx:
    """All sampled values for one note."""

    rng: random.Random
    scenario: dict
    first: str
    last: str
    female: bool
    dob: date
    visit: date
    prior: date
    age: int
    mrn: str
    phone: str
    email: str
    address: str
    physician: tuple
    eye: str
    va_od: str
    va_os: str
    iop: str

    @property
    def full(self) -> str:
        return "%s %s" % (self.first, self.last)

    @property
    def age_text(self) -> str:
        return _phi("AGE_OVER_89", str(self.age)) if self.age > 89 else str(self.age)

    @property
    def sex_word(self) -> str:
        return "woman" if self.female else "man"

    @property
    def pronoun(self) -> str:
        return "She" if self.female else "He"

    @property
    def title(self) -> str:
        return "Ms." if self.female else "Mr."

    def fields(self) -> dict:
        rng, scenario = self.rng, self.scenario
        fellow = "OS" if self.eye == "OD" else "OD"
        values = {
            "OD": _keep("OD"),
            "OS": _keep("OS"),
            "OU": _keep("OU"),
            "eye": _keep(self.eye),
            "fellow": _keep(fellow),
            "r_thin": rng.randint(58, 76),
            "r_ok": rng.randint(84, 104),
            "r_ok2": rng.randint(84, 104),
            "cst_wet": rng.randint(380, 560),
            "cst_a": rng.randint(240, 310),
            "cst_b": rng.randint(240, 310),
            "cst_thin": rng.randint(150, 210),
            "cst_thin2": rng.randint(150, 210),
            "cct_thick": rng.randint(620, 690),
            "cct_ok": rng.randint(560, 605),
        }
        for index, (name, _dosing) in enumerate(scenario["meds"]):
            values["m%d" % index] = _keep(name)
        for index, eponym in enumerate(scenario["eponyms"]):
            values["epo" if index == 0 else "epo%d" % (index + 1)] = _keep(eponym)
        return values

    def fill(self, key: str) -> str:
        return self.scenario[key].format(**self._fields)

    def meds_list(self) -> list[str]:
        return [
            "%s %s" % (_keep(name), dosing.format(**self._fields))
            for name, dosing in self.scenario["meds"]
        ]

    def __post_init__(self):
        self._fields = self.fields()


def _context(rng: random.Random, over89: bool) -> _Ctx:
    candidates = [s for s in _SCENARIOS if s["ages"][1] >= 90] if over89 else list(_SCENARIOS)
    scenario = rng.choice(candidates)
    visit = date(2025, 1, 6) + timedelta(days=rng.randint(0, 630))
    low, high = (90, 97) if over89 else (scenario["ages"][0], min(scenario["ages"][1], 89))
    target_age = rng.randint(low, high)
    dob = date(visit.year - target_age, rng.randint(1, 12), rng.randint(1, 28))
    age = _age_on(dob, visit)
    if over89 and age < 90:
        dob = dob.replace(year=dob.year - 1)
        age = _age_on(dob, visit)
    female = rng.random() < 0.5
    first = rng.choice(_FEMALE_FIRST if female else _MALE_FIRST)
    last = rng.choice(_PATIENT_LAST)
    iop_low = 16 if "elevated IOP" in scenario["hpi"] else 10
    eye = rng.choice(("OD", "OS"))
    va_eye = rng.choice(scenario["acuity"])
    va_fellow = rng.choice(scenario.get("fellow_acuity", _GOOD_ACUITY))
    va_od, va_os = (va_eye, va_fellow) if eye == "OD" else (va_fellow, va_eye)
    return _Ctx(
        rng=rng,
        scenario=scenario,
        first=first,
        last=last,
        female=female,
        dob=dob,
        visit=visit,
        prior=visit - timedelta(days=rng.randint(45, 240)),
        age=age,
        mrn=str(rng.randint(10_000_000, 99_999_999)),
        phone=_phone(rng),
        email="%s%s%d@example.com" % (first[0].lower(), last.lower().replace("-", ""), rng.randint(10, 99)),
        address=_address(rng, suite=False),
        physician=rng.choice(_PHYSICIANS),
        eye=eye,
        va_od=va_od,
        va_os=va_os,
        iop="%d/%d" % (rng.randint(iop_low, iop_low + 12), rng.randint(iop_low, iop_low + 12)),
    )


# --- Note templates ------------------------------------------------------------------------


def _exam_note(ctx: _Ctx) -> str:
    rng = ctx.rng
    doc_first, doc_last = ctx.physician
    doc_full = "%s %s" % (doc_first, doc_last)
    meds = ctx.meds_list()
    tonometry = rng.choice(_TONOMETRY)
    tonometry = _keep(tonometry) if tonometry == "Goldmann" else tonometry

    if rng.random() < 0.5:
        header = (
            "OPHTHALMOLOGY CLINIC NOTE\n"
            f"Patient: {_phi('PATIENT_NAME', ctx.full)}   MRN: {_phi('MRN', ctx.mrn)}   DOB: {_phi('DATE', _us(ctx.dob))}\n"
            f"Date of service: {_phi('DATE', _us(ctx.visit))}   Attending: {_phi('PROVIDER_NAME', doc_full)}, MD\n"
        )
        hpi_lead = f"{ctx.age_text}-year-old {ctx.sex_word} with"
    else:
        header = (
            f"Pt: {_phi('PATIENT_NAME', '%s, %s' % (ctx.last, ctx.first))} | MRN#{_phi('MRN', ctx.mrn)} | "
            f"DOB {_phi('DATE', _us(ctx.dob))} | DOS {_phi('DATE', _us(ctx.visit))}\n"
            f"Provider: Dr. {_phi('PROVIDER_NAME', doc_last)}\n"
        )
        hpi_lead = f"{ctx.age_text} yo {'F' if ctx.female else 'M'} with"

    lines = [
        header,
        f"CC: {ctx.fill('cc')}",
        f"HPI: {hpi_lead} {ctx.fill('hpi')} Last seen {_phi('DATE', _us_short(ctx.prior))} by "
        f"Dr. {_phi('PROVIDER_NAME', doc_last)}.",
        "Meds: " + ("; ".join(meds) if meds else "none (ocular)"),
        "",
        "EXAM",
        f"VA cc: {_keep('OD')} {_keep(ctx.va_od)}   {_keep('OS')} {_keep(ctx.va_os)}",
        f"IOP ({tonometry}): {_keep(ctx.iop)} mmHg",
        f"SLE: {ctx.fill('sle')}",
        f"DFE: {ctx.fill('dfe')}",
    ]
    if rng.random() < 0.35:
        model, prefix = rng.choice(_DEVICES)
        serial = "%s-%06d" % (prefix, rng.randint(100000, 999999))
        lines.append(f"OCT ({model}, S/N {_phi('DEVICE_ID', serial)}) reviewed; see imaging report.")
    contact = f"Patient callback {_phi('PHONE', ctx.phone)}"
    if rng.random() < 0.5:
        contact += f"; portal email {_phi('EMAIL', ctx.email)}"
    lines += [
        "",
        "A/P:",
        f"1. {ctx.fill('assessment')}.",
        f"2. RTC {rng.choice(_INTERVALS)}. {contact}.",
        "",
        f"Electronically signed: {_phi('PROVIDER_NAME', doc_full)}, MD {_phi('DATE', _us(ctx.visit))}",
    ]
    return "\n".join(lines)


def _imaging_report(ctx: _Ctx) -> str:
    rng = ctx.rng
    model, prefix = rng.choice(_DEVICES)
    serial = "%s-%06d" % (prefix, rng.randint(100000, 999999))
    doc_full = "%s %s" % ctx.physician
    reader = "%s %s" % rng.choice([p for p in _PHYSICIANS if p != ctx.physician])
    if rng.random() < 0.5:
        name = "%s^%s" % (ctx.last.upper(), ctx.first.upper())
    else:
        name = "%s, %s" % (ctx.last, ctx.first)
    find_eye, find_fellow = ctx.fill("find_eye"), ctx.fill("find_fellow")
    find_od, find_os = (find_eye, find_fellow) if ctx.eye == "OD" else (find_fellow, find_eye)
    signed = ctx.visit + timedelta(days=rng.randint(0, 3))
    lines = [
        f"OCT REPORT - {model}",
        f"Patient: {_phi('PATIENT_NAME', name)}   ID: {_phi('MRN', ctx.mrn)}   DOB: {_phi('DATE', _iso(ctx.dob))}   "
        f"Sex: {'F' if ctx.female else 'M'}   Age: {ctx.age_text}",
        f"Exam date: {_phi('DATE', _iso(ctx.visit))}   Operator: {_phi('PROVIDER_NAME', rng.choice(_TECHNICIANS))}",
        f"Device S/N: {_phi('DEVICE_ID', serial)}   Referring: Dr. {_phi('PROVIDER_NAME', doc_full)}",
        f"Scan: {ctx.fill('scan')}   Image quality: {rng.randint(45, 78)}/100",
        "",
        "FINDINGS",
        f"{_keep('OD')}: {find_od}",
        f"{_keep('OS')}: {find_os}",
        "",
        f"IMPRESSION: {ctx.fill('impression')}. Compared with {_phi('DATE', _iso(ctx.prior))}: "
        f"{ctx.scenario['change']}.",
        "",
        f"Read by {_phi('PROVIDER_NAME', reader)}, MD on {_phi('DATE', _iso(signed))}",
    ]
    return "\n".join(lines)


def _referral_letter(ctx: _Ctx) -> str:
    rng = ctx.rng
    if rng.random() < 0.7:
        sender, credential = rng.choice(_OPTOMETRISTS), "OD"
    else:
        sender, credential = rng.choice([p for p in _PHYSICIANS if p != ctx.physician]), "MD"
    sender_practice, sender_slug = rng.choice(_REFERRING_PRACTICES)
    recipient_practice = _SPECIALTY_PRACTICES[ctx.scenario["specialty"]]
    recipient_first, recipient_last = ctx.physician
    letter_date = ctx.visit + timedelta(days=rng.randint(0, 3))
    meds = [_keep(name) for name, _ in ctx.scenario["meds"]]
    if meds:
        meds_sentence = f"{ctx.pronoun} currently uses {' and '.join(meds)}."
    else:
        meds_sentence = f"{ctx.pronoun} takes no ocular medications."
    contact = _phi("PHONE", ctx.phone)
    if rng.random() < 0.4:
        contact += f" or {_phi('EMAIL', ctx.email)}"
    re_line = (
        f"RE: {_phi('PATIENT_NAME', ctx.full)}   DOB: {_phi('DATE', _us(ctx.dob))}   MRN: {_phi('MRN', ctx.mrn)}"
    )
    if rng.random() < 0.5:
        re_line += f"\n    {_phi('ADDRESS', ctx.address)}"
    lines = [
        sender_practice,
        _phi("ADDRESS", _address(rng, suite=True)),
        f"Tel {_phi('PHONE', _phone(rng))} | {_phi('EMAIL', 'referrals@%s.example.org' % sender_slug)}",
        "",
        _phi("DATE", _long(letter_date)),
        "",
        f"{_phi('PROVIDER_NAME', '%s %s' % (recipient_first, recipient_last))}, MD",
        recipient_practice,
        _phi("ADDRESS", _address(rng, suite=rng.random() < 0.5)),
        "",
        re_line,
        "",
        f"Dear Dr. {_phi('PROVIDER_NAME', recipient_last)},",
        "",
        f"Thank you for seeing {_phi('PATIENT_NAME', ctx.full)}, a {ctx.age_text}-year-old {ctx.sex_word} I examined on "
        f"{_phi('DATE', _long(ctx.visit))} for {ctx.fill('reason')}. Best-corrected acuity was {_keep(ctx.va_od)} "
        f"{_keep('OD')} and {_keep(ctx.va_os)} {_keep('OS')}, and IOP was {_keep(ctx.iop)} mmHg. {meds_sentence}",
        "",
        f"I would appreciate your evaluation for {ctx.fill('request')}. {ctx.title} {_phi('PATIENT_NAME', ctx.last)} "
        f"can be reached at {contact}. Records and imaging are enclosed.",
        "",
        "Sincerely,",
        "",
        f"{_phi('PROVIDER_NAME', '%s %s' % sender)}, {credential}",
        sender_practice,
    ]
    return "\n".join(lines)


_BUILDERS = {
    "exam_note": _exam_note,
    "imaging_report": _imaging_report,
    "referral_letter": _referral_letter,
}


# --- Generation and integrity ----------------------------------------------------------------


def check_record(record: dict) -> list[str]:
    """Return integrity problems: bad offsets, unlabeled PHI repeats, misplaced protected values."""
    text = record["source_text"]
    spans = record["phi_spans"]
    problems = []
    ranges = [(s["start"], s["end"]) for s in spans]
    for span in spans:
        if text[span["start"]:span["end"]] != span["text"]:
            problems.append("offset mismatch for %r" % span["text"])
        for start, end in occurrences(span["text"], text):
            if not any(a <= start and end <= b for a, b in ranges):
                problems.append("unlabeled occurrence of %r at %d" % (span["text"], start))
    for value in record["protected_values"]:
        found = occurrences(value, text)
        if not found:
            problems.append("protected value %r not in text" % value)
        for start, end in found:
            if any(start < b and a < end for a, b in ranges):
                problems.append("protected value %r overlaps PHI at %d" % (value, start))
    return problems


def generate(seed: int = DEFAULT_SEED, size: int = DEFAULT_SIZE) -> list[dict]:
    rng = random.Random(seed)
    note_types = [NOTE_TYPES[i % len(NOTE_TYPES)] for i in range(size)]
    rng.shuffle(note_types)
    records = []
    for index, note_type in enumerate(note_types):
        over89 = index % 5 == 2
        for _attempt in range(200):
            text, spans, protected = _render(_BUILDERS[note_type](_context(rng, over89)))
            record = {
                "id": "note-%03d" % (index + 1),
                "note_type": note_type,
                "source_text": text,
                "phi_spans": spans,
                "protected_values": protected,
            }
            if not check_record(record):
                break
        else:
            raise RuntimeError("could not generate a consistent %s note" % note_type)
        records.append(record)
    return records


def write_dataset(path: Path, records: list[dict]) -> None:
    write_jsonl(path, records)
