"""Hugging Face medical NER and symptom-to-feature mapping.

This module deliberately performs language understanding only. It never assigns a
disease label: the final prediction is made by the persisted scikit-learn/XGBoost
model in :mod:`src.predictor`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

try:
    from rapidfuzz import fuzz, process
except ImportError:  # optional dependency
    fuzz = process = None

from .config import (
    MEDICAL_NER_MODEL,
    MEDICAL_SEMANTIC_MODEL,
    MEDICAL_SEMANTIC_THRESHOLD,
    MEDICAL_SEMANTIC_TOP_K,
)
from .data_pipeline import normalize_key


# Everyday wording is mapped only to an existing Training.csv feature. This keeps
# free text convenient without creating any symptom or disease outside the data.
#
# Keys are colloquial plain-English phrasings; values must be exact Training.csv
# feature names (underscores). The mapper normalises both sides before comparison,
# so case and punctuation in the key strings do not matter.
COMMON_ALIASES = {
    "stiff joints": "joint pain",
    "joints are stiff": "joint pain",
    # ── abdominal / stomach ────────────────────────────────────────────────────
    "stomach ache": "stomach_pain",
    "stomachache": "stomach_pain",
    "tummy ache": "stomach_pain",
    "tummy pain": "stomach_pain",
    "stomach hurts": "stomach_pain",
    "stomach hurting": "stomach_pain",
    "gut pain": "stomach_pain",
    "gut ache": "stomach_pain",
    "gut hurts": "stomach_pain",
    "gastric": "stomach_pain",
    "gastritis pain": "stomach_pain",
    "abdominal pain": "abdominal_pain",
    "belly pain": "belly_pain",
    "belly ache": "belly_pain",
    "belly hurts": "belly_pain",
    "lower abdomen pain": "abdominal_pain",
    "lower belly pain": "abdominal_pain",
    "upper abdomen pain": "abdominal_pain",
    "upper stomach pain": "abdominal_pain",
    "lower stomach pain": "abdominal_pain",
    "side pain": "abdominal_pain",
    "flank pain": "abdominal_pain",
    "pelvic pain": "abdominal_pain",
    "abdominal cramps": "cramps",
    "stomach cramps": "cramps",
    "belly cramps": "cramps",
    "gut cramps": "cramps",
    "intestinal cramps": "cramps",
    "bowel cramps": "cramps",
    "crampy belly": "cramps",
    "cramping belly": "cramps",
    "cramping stomach": "cramps",
    "menstrual cramps": "cramps",
    "period cramps": "cramps",
    "period pain": "cramps",
    "menstrual pain": "cramps",
    "dysmenorrhea": "cramps",
    "mid-cycle pain": "cramps",
    "bloating": "distention_of_abdomen",
    "bloated stomach": "distention_of_abdomen",
    "bloated belly": "distention_of_abdomen",
    "distended abdomen": "distention_of_abdomen",
    "swollen belly": "swelling_of_stomach",
    "swollen stomach": "swelling_of_stomach",
    "gas": "passage_of_gases",
    "flatulence": "passage_of_gases",
    "passing gas": "passage_of_gases",
    "gassy": "passage_of_gases",
    "gas pains": "passage_of_gases",
    "gas pain": "passage_of_gases",
    "painful gas": "passage_of_gases",
    "heartburn": "acidity",
    "acid reflux": "acidity",
    "reflux": "acidity",
    "sour burp": "acidity",
    "sour stomach": "acidity",
    "acid burp": "acidity",
    "acid taste": "acidity",
    "acid coming up": "acidity",
    "burning stomach": "acidity",
    "burning belly": "acidity",
    "gnawing pain": "acidity",
    "gnawing stomach": "acidity",
    "indigestion": "indigestion",
    "upset stomach": "indigestion",
    "tummy upset": "indigestion",
    "stomach upset": "indigestion",
    "stomach bug": "indigestion",
    "tummy bug": "indigestion",
    "gut trouble": "indigestion",
    "gut issues": "indigestion",
    "gut feeling bad": "indigestion",
    "gut feeling unwell": "indigestion",
    "constipated": "constipation",
    "constipation": "constipation",
    "can't poop": "constipation",
    "hard stool": "constipation",
    "hard stools": "constipation",
    "dry stool": "constipation",
    "pellet stool": "constipation",
    "rabbit pellets": "constipation",
    "constipation pain": "constipation",
    "difficulty pooping": "constipation",
    "difficulty passing stool": "constipation",
    "can't pass stool": "constipation",
    "diarrhea": "diarrhoea",
    "diarrhoea": "diarrhoea",
    "loose stools": "diarrhoea",
    "loose motions": "diarrhoea",
    "watery stool": "diarrhoea",
    "runny stool": "diarrhoea",
    "runny poop": "diarrhoea",
    "loose poop": "diarrhoea",
    "frequent loose stools": "diarrhoea",
    "bloody stool": "bloody_stool",
    "blood in stool": "bloody_stool",
    "bleeding stool": "bloody_stool",
    "blood when wiping": "bloody_stool",
    "blood on toilet paper": "bloody_stool",
    "rectal bleeding": "bloody_stool",
    "bright red blood in stool": "bloody_stool",
    "rectal pain": "pain_in_anal_region",
    "anal pain": "pain_in_anal_region",
    "pain in rectum": "pain_in_anal_region",
    "butt pain": "pain_in_anal_region",
    "pain around anus": "pain_in_anal_region",
    "pain during bowel movement": "pain_during_bowel_movements",
    "painful bowel movement": "pain_during_bowel_movements",
    "painful poop": "pain_during_bowel_movements",
    "painful defecation": "pain_during_bowel_movements",
    "pain when pooping": "pain_during_bowel_movements",
    "rectal itching": "irritation_in_anus",
    "anal itching": "irritation_in_anus",
    "itchy anus": "irritation_in_anus",
    "itchy butt": "irritation_in_anus",
    "rectal itch": "irritation_in_anus",
    "stomach bleeding": "stomach_bleeding",
    "internal bleeding": "stomach_bleeding",
    "vomiting blood": "stomach_bleeding",
    "blood in vomit": "stomach_bleeding",
    # ── fever / temperature ────────────────────────────────────────────────────
    "fever": "high_fever",
    "high fever": "high_fever",
    "high temperature": "high_fever",
    "temperature": "mild_fever",
    "slight fever": "mild_fever",
    "low grade fever": "mild_fever",
    "mild fever": "mild_fever",
    "feeling hot": "high_fever",
    "burning up": "high_fever",
    "boiling hot": "high_fever",
    "roasting": "high_fever",
    "very hot": "high_fever",
    "extremely hot": "high_fever",
    "feverish": "high_fever",
    "febrile": "high_fever",
    "pyrexia": "high_fever",
    "body hot": "high_fever",
    "temperature is high": "high_fever",
    "temperature up": "high_fever",
    "temp up": "high_fever",
    "high temp": "high_fever",
    "chills": "chills",
    "shivering": "shivering",
    "shivering a lot": "shivering",
    "can't stop shivering": "shivering",
    "teeth chattering": "chills",
    "rigors": "shivering",
    "feeling cold": "chills",
    "freezing": "chills",
    "freezing cold": "chills",
    "can't get warm": "chills",
    "cold and hot": "chills",
    "alternating hot and cold": "chills",
    "hot then cold": "chills",
    "cold sweat": "sweating",
    "cold sweats": "sweating",
    "sweating": "sweating",
    "sweating a lot": "sweating",
    "sweating then chills": "sweating",
    "night sweats": "sweating",
    "excessive sweating": "sweating",
    "drenched in sweat": "sweating",
    "sweaty": "sweating",
    "perspiring": "sweating",
    "perspire": "sweating",
    "clammy": "sweating",
    "clammy skin": "sweating",
    "hot flushes": "sweating",
    "hot flush": "sweating",
    "hot flashes": "sweating",
    "hot flash": "sweating",
    # ── head / neurological ────────────────────────────────────────────────────
    "headache": "headache",
    "head pain": "headache",
    "head hurts": "headache",
    "head hurting": "headache",
    "head ache": "headache",
    "migraine": "headache",
    "head throbbing": "headache",
    "throbbing head": "headache",
    "throbbing headache": "headache",
    "splitting headache": "headache",
    "pounding head": "headache",
    "pounding headache": "headache",
    "pressure in head": "headache",
    "head pressure": "headache",
    "head feels heavy": "headache",
    "heavy head": "headache",
    "skull pain": "headache",
    "pain behind eyes": "pain_behind_the_eyes",
    "pain behind the eyes": "pain_behind_the_eyes",
    "behind the eyes": "pain_behind_the_eyes",
    "eye pain": "pain_behind_the_eyes",
    "eyes hurt": "pain_behind_the_eyes",
    "eye socket pain": "pain_behind_the_eyes",
    "pain in eye socket": "pain_behind_the_eyes",
    "vision pain": "pain_behind_the_eyes",
    "dizziness": "dizziness",
    "dizzy": "dizziness",
    "feeling dizzy": "dizziness",
    "light headed": "dizziness",
    "lightheaded": "dizziness",
    "dizzy spells": "dizziness",
    "dizzy when standing": "dizziness",
    "dizzy when moving": "dizziness",
    "vertigo": "spinning_movements",
    "spinning sensation": "spinning_movements",
    "room spinning": "spinning_movements",
    "head spinning": "spinning_movements",
    "world spinning": "spinning_movements",
    "everything spinning": "spinning_movements",
    "feeling faint": "dizziness",
    "faint": "dizziness",
    "about to faint": "dizziness",
    "loss of balance": "loss_of_balance",
    "balance problems": "loss_of_balance",
    "balance off": "loss_of_balance",
    "balance gone": "loss_of_balance",
    "unsteady": "unsteadiness",
    "unsteadiness": "unsteadiness",
    "poor balance": "unsteadiness",
    "tripping": "unsteadiness",
    "falling over": "unsteadiness",
    "confusion": "altered_sensorium",
    "confused": "altered_sensorium",
    "disoriented": "altered_sensorium",
    "mental confusion": "altered_sensorium",
    "can't think straight": "altered_sensorium",
    "foggy brain": "altered_sensorium",
    "brain fog": "altered_sensorium",
    "foggy head": "altered_sensorium",
    "fuzzy head": "altered_sensorium",
    "mental fog": "altered_sensorium",
    "memory loss": "lack_of_concentration",
    "forgetfulness": "lack_of_concentration",
    "forgetful": "lack_of_concentration",
    "absent minded": "lack_of_concentration",
    "absent-minded": "lack_of_concentration",
    "losing track": "lack_of_concentration",
    "poor concentration": "lack_of_concentration",
    "can't concentrate": "lack_of_concentration",
    "concentration issues": "lack_of_concentration",
    "memory issues": "lack_of_concentration",
    "memory problems": "lack_of_concentration",
    "slurred speech": "slurred_speech",
    "difficulty speaking": "slurred_speech",
    "speech problems": "slurred_speech",
    "slurring words": "slurred_speech",
    "speech slurred": "slurred_speech",
    "speech slurry": "slurred_speech",
    "can't talk properly": "slurred_speech",
    "mixing up words": "slurred_speech",
    "can't find words": "slurred_speech",
    "can't remember words": "slurred_speech",
    "numbness": "weakness_in_limbs",
    "tingling": "weakness_in_limbs",
    "numbness in limbs": "weakness_in_limbs",
    "numbness in fingers": "weakness_in_limbs",
    "numbness in toes": "weakness_in_limbs",
    "numbness in hands": "weakness_in_limbs",
    "numbness in feet": "weakness_in_limbs",
    "tingling fingers": "weakness_in_limbs",
    "tingling toes": "weakness_in_limbs",
    "tingling hands": "weakness_in_limbs",
    "tingling feet": "weakness_in_limbs",
    "pins and needles": "weakness_in_limbs",
    "weakness one side": "weakness_of_one_body_side",
    "one sided weakness": "weakness_of_one_body_side",
    "paralysis one side": "weakness_of_one_body_side",
    "one side weak": "weakness_of_one_body_side",
    "half of body weak": "weakness_of_one_body_side",
    "face drooping": "weakness_of_one_body_side",
    "facial droop": "weakness_of_one_body_side",
    "can't smile straight": "weakness_of_one_body_side",
    "can't lift arm": "weakness_of_one_body_side",
    # ── respiratory ───────────────────────────────────────────────────────────
    "cough": "cough",
    "coughing": "cough",
    "dry cough": "cough",
    "wet cough": "cough",
    "persistent cough": "cough",
    "constant cough": "cough",
    "hacking cough": "cough",
    "barking cough": "cough",
    "cough at night": "cough",
    "night cough": "cough",
    "morning cough": "cough",
    "cough in morning": "cough",
    "coughing fits": "cough",
    "coughing spells": "cough",
    "can't stop coughing": "cough",
    "shortness of breath": "shortness of breath",
    "difficulty breathing": "shortness of breath",
    "trouble breathing": "shortness of breath",
    "having trouble breathing": "shortness of breath",
    "can't breathe": "shortness of breath",
    "cannot breathe": "shortness of breath",
    "breathing difficulty": "shortness of breath",
    "breathing problems": "shortness of breath",
    "wheezing": "breathlessness",
    "wheezy": "breathlessness",
    "wheezing sound": "breathlessness",
    "whistle in chest": "breathlessness",
    "whistle when breathing": "breathlessness",
    "gasping": "breathlessness",
    "panting": "breathlessness",
    "breathing fast": "breathlessness",
    "fast breathing": "breathlessness",
    "shallow breathing": "breathlessness",
    "labored breathing": "breathlessness",
    "laboured breathing": "breathlessness",
    "respiratory distress": "breathlessness",
    "can't catch breath": "shortness of breath",
    "can't catch my breath": "shortness of breath",
    "catch my breath": "shortness of breath",
    "out of breath": "shortness of breath",
    "breathlessness at night": "breathlessness",
    "breathlessness when lying down": "breathlessness",
    "breathlessness on exertion": "breathlessness",
    "breathlessness when walking": "breathlessness",
    "stuffy nose": "congestion",
    "blocked nose": "congestion",
    "nasal congestion": "congestion",
    "nose blocked": "congestion",
    "nose stuffed": "congestion",
    "nose bunged": "nasal congestion",
    "nose bunged up": "nasal congestion",
    "nose is bunged up": "nasal congestion",
    "my nose is bunged up": "nasal congestion",
    "bunged up nose": "nasal congestion",
    "stuffed up nose": "congestion",
    "nose stuffed up": "congestion",
    "nose is stuffed up": "congestion",
    "can't breathe through nose": "congestion",
    "runny nose": "runny nose",
    "sneezing": "continuous_sneezing",
    "constant sneezing": "continuous_sneezing",
    "non stop sneezing": "continuous_sneezing",
    "sneezing a lot": "continuous_sneezing",
    "sneezing fits": "continuous_sneezing",
    "phlegm": "phlegm",
    "mucus": "phlegm",
    "sputum": "phlegm",
    "thick mucus": "phlegm",
    "green mucus": "phlegm",
    "yellow mucus": "phlegm",
    "productive cough": "phlegm",
    "coughing up mucus": "phlegm",
    "coughing up phlegm": "phlegm",
    "mucoid sputum": "mucoid_sputum",
    "blood in sputum": "blood_in_sputum",
    "coughing blood": "blood_in_sputum",
    "coughing up blood": "blood_in_sputum",
    "blood when coughing": "blood_in_sputum",
    "taste of blood": "blood_in_sputum",
    "rusty sputum": "rusty_sputum",
    "sinus pressure": "sinus_pressure",
    "sinus pain": "sinus_pressure",
    "sinus headache": "sinus_pressure",
    "sinus": "sinus_pressure",
    "cheek pain": "sinus_pressure",
    "between eyes pain": "sinus_pressure",
    "forehead pressure": "sinus_pressure",
    "post-nasal drip": "sinus_pressure",
    "throat irritation": "throat_irritation",
    "scratchy throat": "throat_irritation",
    "drip in throat": "throat_irritation",
    "sore throat": "patches_in_throat",
    "throat pain": "patches_in_throat",
    "painful swallow": "patches_in_throat",
    "painful swallowing": "patches_in_throat",
    "can't swallow": "patches_in_throat",
    "trouble swallowing": "patches_in_throat",
    "difficulty swallowing": "patches_in_throat",
    "hoarse": "patches_in_throat",
    "hoarseness": "patches_in_throat",
    "lost voice": "patches_in_throat",
    "voice gone": "patches_in_throat",
    "patches in throat": "patches_in_throat",
    # ── eyes ──────────────────────────────────────────────────────────────────
    "red eyes": "redness_of_eyes",
    "bloodshot eyes": "redness_of_eyes",
    "pink eye": "redness_of_eyes",
    "eye redness": "redness_of_eyes",
    "red eye": "redness_of_eyes",
    "itchy eyes": "redness_of_eyes",
    "scratchy eyes": "redness_of_eyes",
    "gritty eyes": "redness_of_eyes",
    "sand in eyes": "redness_of_eyes",
    "blurry vision": "blurred_and_distorted_vision",
    "blurred vision": "blurred_and_distorted_vision",
    "blurry": "blurred_and_distorted_vision",
    "can't see clearly": "blurred_and_distorted_vision",
    "vision blurred": "blurred_and_distorted_vision",
    "vision fuzzy": "blurred_and_distorted_vision",
    "can't focus vision": "blurred_and_distorted_vision",
    "vision problems": "blurred_and_distorted_vision",
    "poor vision": "blurred_and_distorted_vision",
    "see spots": "visual_disturbances",
    "spots in vision": "visual_disturbances",
    "see floaters": "visual_disturbances",
    "halos around lights": "visual_disturbances",
    "light sensitivity": "visual_disturbances",
    "sensitive to light": "visual_disturbances",
    "bright light hurts": "visual_disturbances",
    "photophobia": "visual_disturbances",
    "dark spots": "visual_disturbances",
    "black spots": "visual_disturbances",
    "visual disturbances": "visual_disturbances",
    "watery eyes": "watering_from_eyes",
    "teary eyes": "watering_from_eyes",
    "eyes watering": "watering_from_eyes",
    "tears running": "watering_from_eyes",
    "tears dripping": "watering_from_eyes",
    "tearing": "watering_from_eyes",
    "puffy eyes": "puffy_face_and_eyes",
    "swollen eyes": "puffy_face_and_eyes",
    "puffy eyelids": "puffy_face_and_eyes",
    "swollen eyelids": "puffy_face_and_eyes",
    "swollen around eyes": "puffy_face_and_eyes",
    "puffy face": "puffy_face_and_eyes",
    "swollen face": "puffy_face_and_eyes",
    "yellow eyes": "yellowing_of_eyes",
    "jaundiced eyes": "yellowing_of_eyes",
    "yellow sclera": "yellowing_of_eyes",
    "yellow whites of eyes": "yellowing_of_eyes",
    # ── skin ──────────────────────────────────────────────────────────────────
    "skin rash": "skin_rash",
    "rash": "skin_rash",
    "rashes": "skin_rash",
    "hives": "skin_rash",
    "welts": "skin_rash",
    "urticaria": "skin_rash",
    "flat rash": "skin_rash",
    "itchy rash": "skin_rash",
    "burning rash": "skin_rash",
    "painful rash": "skin_rash",
    "rash on chest": "skin_rash",
    "rash on back": "skin_rash",
    "rash on arms": "skin_rash",
    "rash on legs": "skin_rash",
    "itchy bumps": "skin_rash",
    "itching": "itching",
    "itchy skin": "itching",
    "scratching": "itching",
    "scratching a lot": "itching",
    "scratching at night": "itching",
    "skin itch": "itching",
    "burning skin": "itching",
    "tingling skin": "itching",
    "numb skin": "itching",
    "painful skin": "itching",
    "sensitive skin": "itching",
    "yellow skin": "yellowish_skin",
    "yellowish skin": "yellowish_skin",
    "jaundice": "yellowish_skin",
    "jaundiced": "yellowish_skin",
    "skin turned yellow": "yellowish_skin",
    "orange skin": "yellowish_skin",
    "skin peeling": "skin_peeling",
    "peeling skin": "skin_peeling",
    "flaky skin": "skin_peeling",
    "flaking": "skin_peeling",
    "dry skin peeling": "skin_peeling",
    "scaly": "skin_peeling",
    "scaly patches": "skin_peeling",
    "blackheads": "blackheads",
    "pimples": "pus_filled_pimples",
    "pus pimples": "pus_filled_pimples",
    "acne": "pus_filled_pimples",
    "blisters": "blister",
    "blister": "blister",
    "fluid filled blister": "blister",
    "skin spots": "red_spots_over_body",
    "red spots": "red_spots_over_body",
    "spots on skin": "red_spots_over_body",
    "spots on body": "red_spots_over_body",
    "red dots": "red_spots_over_body",
    "red dots on skin": "red_spots_over_body",
    "rash with bumps": "nodal_skin_eruptions",
    "raised rash": "nodal_skin_eruptions",
    "nodular rash": "nodal_skin_eruptions",
    "bumpy rash": "nodal_skin_eruptions",
    "skin eruptions": "nodal_skin_eruptions",
    "skin bumps": "nodal_skin_eruptions",
    "bruising": "bruising",
    "bruises": "bruising",
    "easy bruising": "bruising",
    "bruise easily": "bruising",
    "bruising easily": "bruising",
    "discolored patches": "dischromic _patches",
    "skin discoloration": "dischromic _patches",
    "skin discoloured": "dischromic _patches",
    "patches on skin": "dischromic _patches",
    "dark patches on skin": "dischromic _patches",
    "light patches on skin": "dischromic _patches",
    "silver patches": "silver_like_dusting",
    "scaly skin": "silver_like_dusting",
    "silver like dusting": "silver_like_dusting",
    "silver dust on skin": "silver_like_dusting",
    "dry cracked lips": "drying_and_tingling_lips",
    "tingling lips": "drying_and_tingling_lips",
    "cracked lips": "drying_and_tingling_lips",
    "dry lips": "drying_and_tingling_lips",
    "chapped lips": "drying_and_tingling_lips",
    "sores around nose": "red_sore_around_nose",
    "sore near nose": "red_sore_around_nose",
    "sore by nose": "red_sore_around_nose",
    "yellow crust": "yellow_crust_ooze",
    "skin ooze": "yellow_crust_ooze",
    "crust on skin": "yellow_crust_ooze",
    "crusty ooze": "yellow_crust_ooze",
    "oozing crust": "yellow_crust_ooze",
    # ── joints / muscles / movement ───────────────────────────────────────────
    "joint pain": "joint pain",
    "joint ache": "joint_pain",
    "aching joints": "joint_pain",
    "joints hurt": "joint_pain",
    "joints ache": "joint_pain",
    "painful joints": "joint_pain",
    "joint stiffness": "swelling_joints",
    "joint swelling": "swelling_joints",
    "swollen joints": "swelling_joints",
    "stiff joints": "joint pain",
    "joints are stiff": "joint pain",
    "joints feel stiff": "swelling_joints",
    "knee pain": "knee_pain",
    "knee ache": "knee_pain",
    "knee hurts": "knee_pain",
    "knees hurt": "knee_pain",
    "knees ache": "knee_pain",
    "knee swelling": "knee_pain",
    "knee swollen": "knee_pain",
    "knee giving out": "knee_pain",
    "hip pain": "hip_joint_pain",
    "hip ache": "hip_joint_pain",
    "hip hurts": "hip_joint_pain",
    "hip stiff": "hip_joint_pain",
    "back pain": "back_pain",
    "lower back pain": "back_pain",
    "upper back pain": "back_pain",
    "mid back pain": "back_pain",
    "backache": "back_pain",
    "sciatica": "back_pain",
    "sciatic pain": "back_pain",
    "neck pain": "neck_pain",
    "neck ache": "neck_pain",
    "stiff neck": "stiff_neck",
    "neck stiffness": "stiff_neck",
    "can't turn neck": "stiff_neck",
    "neck won't turn": "stiff_neck",
    "muscle pain": "muscle_pain",
    "muscle ache": "muscle_pain",
    "sore muscles": "muscle_pain",
    "muscles sore": "muscle_pain",
    "muscles hurt": "muscle_pain",
    "body ache": "muscle_pain",
    "body aches": "muscle_pain",
    "body pain": "muscle_pain",
    "hurts all over": "muscle_pain",
    "muscle weakness": "muscle_weakness",
    "weak muscles": "muscle_weakness",
    "muscles weak": "muscle_weakness",
    "can't lift": "muscle_weakness",
    "can't grip": "muscle_weakness",
    "can't squeeze": "muscle_weakness",
    "grip weak": "muscle_weakness",
    "muscle twitching": "muscle_pain",
    "muscle spasms": "muscle_pain",
    "muscle cramps": "muscle_pain",
    "charley horse": "muscle_pain",
    "leg cramps": "muscle_pain",
    "foot cramps": "muscle_pain",
    "calf cramps": "muscle_pain",
    "muscle wasting": "muscle_wasting",
    "muscle loss": "muscle_wasting",
    "movement stiffness": "movement_stiffness",
    "stiff movement": "movement_stiffness",
    "stiff body": "movement_stiffness",
    "stiff all over": "movement_stiffness",
    "difficulty walking": "painful_walking",
    "painful walking": "painful_walking",
    "walking hurts": "painful_walking",
    "limping": "painful_walking",
    "pain when walking": "painful_walking",
    "can't walk properly": "painful_walking",
    "can't climb stairs": "painful_walking",
    "stairs hurt": "painful_walking",
    "cold hands": "cold_hands_and_feets",
    "cold feet": "cold_hands_and_feets",
    "cold fingers": "cold_hands_and_feets",
    "cold toes": "cold_hands_and_feets",
    "cold extremities": "cold_hands_and_feets",
    "swollen legs": "swollen_legs",
    "leg swelling": "swollen_legs",
    "swollen ankles": "swollen_legs",
    "swollen wrists": "swollen_legs",
    "swollen extremities": "swollen_extremeties",
    "swollen hands": "swollen_extremeties",
    "swollen feet": "swollen_extremeties",
    "prominent veins": "prominent_veins_on_calf",
    "varicose veins": "prominent_veins_on_calf",
    "visible veins on leg": "prominent_veins_on_calf",
    "veins bulging": "prominent_veins_on_calf",
    "veins sticking out": "prominent_veins_on_calf",
    "spider veins on legs": "prominent_veins_on_calf",
    # ── urinary ───────────────────────────────────────────────────────────────
    "burning when urinating": "burning_micturition",
    "burning while urinating": "burning_micturition",
    "painful urination": "burning_micturition",
    "pain urinating": "burning_micturition",
    "pain peeing": "burning_micturition",
    "hurts to pee": "burning_micturition",
    "burning pee": "painful urination",
    "stinging pee": "burning_micturition",
    "stinging when peeing": "burning_micturition",
    "dysuria": "burning_micturition",
    "frequent urination": "polyuria",
    "peeing a lot": "polyuria",
    "urinating often": "polyuria",
    "excessive urination": "polyuria",
    "increased urination": "polyuria",
    "going to bathroom a lot": "polyuria",
    "dark urine": "dark_urine",
    "brown urine": "dark_urine",
    "tea colored urine": "dark_urine",
    "urine dark": "dark_urine",
    "yellow urine": "yellow_urine",
    "foul smelling urine": "foul_smell_of urine",
    "smelly urine": "foul_smell_of urine",
    "bad smelling urine": "foul_smell_of urine",
    "strong smelling urine": "foul_smell_of urine",
    "urine smells bad": "foul_smell_of urine",
    "urine odor": "foul_smell_of urine",
    "bladder discomfort": "bladder_discomfort",
    "bladder pain": "bladder_discomfort",
    "bladder pressure": "bladder_discomfort",
    "bladder cramps": "bladder_discomfort",
    "bladder spasms": "bladder_discomfort",
    "urge to urinate": "continuous_feel_of_urine",
    "constant urge to urinate": "continuous_feel_of_urine",
    "can't hold urine": "continuous_feel_of_urine",
    "urgency to pee": "continuous_feel_of_urine",
    "sudden urge to pee": "continuous_feel_of_urine",
    "can't stop peeing": "continuous_feel_of_urine",
    "need to pee all the time": "continuous_feel_of_urine",
    "need to urinate": "continuous_feel_of_urine",
    "need to pee": "continuous_feel_of_urine",
    "always feel like peeing": "continuous_feel_of_urine",
    "dribbling": "continuous_feel_of_urine",
    "leaking urine": "continuous_feel_of_urine",
    "incontinence": "continuous_feel_of_urine",
    "can't control bladder": "continuous_feel_of_urine",
    "blood in urine": "bladder_discomfort",
    "pink urine": "bladder_discomfort",
    "urine red": "bladder_discomfort",
    "blood when peeing": "bladder_discomfort",
    "orange urine": "bladder_discomfort",
    "weak stream": "continuous_feel_of_urine",
    "slow stream": "continuous_feel_of_urine",
    "can't empty bladder": "continuous_feel_of_urine",
    "spotting urination": "spotting_ urination",
    # ── fatigue / general ─────────────────────────────────────────────────────
    "tiredness": "fatigue",
    "fatigue": "fatigue",
    "tired": "fatigue",
    "exhaustion": "fatigue",
    "exhausted": "fatigue",
    "drained": "fatigue",
    "worn out": "fatigue",
    "worn-out": "fatigue",
    "run down": "fatigue",
    "run-down": "fatigue",
    "wiped out": "fatigue",
    "no energy": "fatigue",
    "no stamina": "fatigue",
    "can't keep up": "fatigue",
    "can't get going": "fatigue",
    "feeling weak": "fatigue",
    "weakness": "fatigue",
    "feeling blah": "fatigue",
    "feeling flat": "fatigue",
    "lethargy": "lethargy",
    "lethargic": "lethargy",
    "sluggish": "lethargy",
    "slow": "lethargy",
    "can't move": "lethargy",
    "can barely move": "lethargy",
    "moving slowly": "lethargy",
    "malaise": "malaise",
    "feeling unwell": "malaise",
    "feeling under the weather": "malaise",
    "not feeling well": "malaise",
    "not feeling great": "malaise",
    "feeling off": "malaise",
    "off-color": "malaise",
    "off colour": "malaise",
    "generally unwell": "malaise",
    "restlessness": "restlessness",
    "can't sleep": "restlessness",
    "cant sleep": "restlessness",
    "cannot sleep": "restlessness",
    "sleeplessness": "restlessness",
    "poor sleep": "restlessness",
    "insomnia": "restlessness",
    "trouble sleeping": "restlessness",
    "unable to sleep": "restlessness",
    "can't fall asleep": "restlessness",
    "can't stay asleep": "restlessness",
    "waking up at night": "restlessness",
    "early waking": "restlessness",
    "broken sleep": "restlessness",
    "sleep trouble": "restlessness",
    "restless": "restlessness",
    "restless legs": "restlessness",
    "restless sleep": "restlessness",
    "anxious": "anxiety",
    "anxiety": "anxiety",
    "nervousness": "anxiety",
    "nervous": "anxiety",
    "feeling anxious": "anxiety",
    "worried": "anxiety",
    "worried sick": "anxiety",
    "panic": "anxiety",
    "panic attack": "anxiety",
    "depression": "depression",
    "feeling depressed": "depression",
    "depressed": "depression",
    "low mood": "depression",
    "sad": "depression",
    "sad mood": "depression",
    "sadness": "depression",
    "down in the dumps": "depression",
    "blue": "depression",
    "mood swings": "mood_swings",
    "moody": "mood_swings",
    "irritable": "irritability",
    "irritability": "irritability",
    "cranky": "irritability",
    "grumpy": "irritability",
    "easily annoyed": "irritability",
    "easily angered": "irritability",
    "angry": "irritability",
    "dehydration": "dehydration",
    "dehydrated": "dehydration",
    "dry mouth": "dehydration",
    "dry lips": "dehydration",
    "not drinking enough": "dehydration",
    "sunken eyes": "sunken_eyes",
    "hollow eyes": "sunken_eyes",
    "deep-set eyes": "sunken_eyes",
    # ── appetite / weight ─────────────────────────────────────────────────────
    "loss of appetite": "loss_of_appetite",
    "no appetite": "loss_of_appetite",
    "not hungry": "loss_of_appetite",
    "poor appetite": "loss_of_appetite",
    "not eating": "loss_of_appetite",
    "can't eat": "loss_of_appetite",
    "can't finish meal": "loss_of_appetite",
    "food doesn't taste good": "loss_of_appetite",
    "food tastes off": "loss_of_appetite",
    "no desire to eat": "loss_of_appetite",
    "lost appetite": "loss_of_appetite",
    "increased appetite": "increased_appetite",
    "always hungry": "increased_appetite",
    "can't stop eating": "increased_appetite",
    "eating a lot": "increased_appetite",
    "overeating": "increased_appetite",
    "excessive hunger": "excessive_hunger",
    "very hungry": "excessive_hunger",
    "hungry all the time": "excessive_hunger",
    "never hungry": "loss_of_appetite",
    "weight loss": "weight_loss",
    "losing weight": "weight_loss",
    "sudden weight loss": "weight_loss",
    "weight dropping": "weight_loss",
    "weight falling": "weight_loss",
    "weight gain": "weight_gain",
    "gaining weight": "weight_gain",
    "putting on weight": "weight_gain",
    "weight going up": "weight_gain",
    "obesity": "obesity",
    "overweight": "obesity",
    "obese": "obesity",
    "fat": "obesity",
    "bmi high": "obesity",
    # ── cardiovascular ────────────────────────────────────────────────────────
    "heart racing": "fast_heart_rate",
    "fast heartbeat": "fast_heart_rate",
    "rapid heartbeat": "fast_heart_rate",
    "racing heart": "fast_heart_rate",
    "fast heart": "fast_heart_rate",
    "palpitations": "palpitations",
    "heart flutters": "palpitations",
    "heart fluttering": "palpitations",
    "heart pounding": "palpitations",
    "heart skipping": "palpitations",
    "missed beat": "palpitations",
    "extra beat": "palpitations",
    "irregular heartbeat": "palpitations",
    "arrhythmia": "palpitations",
    "heart feels weird": "palpitations",
    "chest pain": "chest_pain",
    "chest hurting": "chest_pain",
    "chest tightness": "chest tightness",
    "chest is tight": "chest_pain",
    "tight chest": "chest_pain",
    "chest feels tight": "chest_pain",
    "chest pressure": "chest_pain",
    "chest discomfort": "chest_pain",
    "chest heaviness": "chest_pain",
    "crushing chest": "chest_pain",
    "squeezing chest": "chest_pain",
    "elephant on chest": "chest_pain",
    "sharp chest pain": "chest_pain",
    "stabbing chest": "chest_pain",
    "chest pain when breathing": "chest_pain",
    "chest pain on exertion": "chest_pain",
    "angina": "chest_pain",
    "swollen blood vessels": "swollen_blood_vessels",
    "bulging veins": "swollen_blood_vessels",
    "fluid overload": "fluid_overload",
    "fluid retention": "fluid_overload",
    "water retention": "fluid_overload",
    "retaining water": "fluid_overload",
    "swelling all over": "fluid_overload",
    "generalized swelling": "fluid_overload",
    # ── nausea / vomiting ─────────────────────────────────────────────────────
    "nausea": "nausea",
    "feeling sick": "nausea",
    "nauseous": "nausea",
    "nauseated": "nausea",
    "queasy": "nausea",
    "sick to stomach": "nausea",
    "sick to my stomach": "nausea",
    "sick in my stomach": "nausea",
    "want to throw up": "nausea",
    "about to vomit": "nausea",
    "morning sickness": "nausea",
    "seasick": "nausea",
    "motion sick": "nausea",
    "carsick": "nausea",
    "vomiting": "vomiting",
    "vomit": "vomiting",
    "throwing up": "vomiting",
    "puking": "vomiting",
    "puke": "vomiting",
    "upchuck": "vomiting",
    "dry heaving": "vomiting",
    "retching": "vomiting",
    "gagging": "vomiting",
    "vomited": "vomiting",
    "brought up food": "vomiting",
    "hematemesis": "vomiting",
    # ── lymph / glands ────────────────────────────────────────────────────────
    "swollen glands": "swelled_lymph_nodes",
    "swollen nodes": "swelled_lymph_nodes",
    "enlarged lymph nodes": "swelled_lymph_nodes",
    "swollen lymph nodes": "swelled_lymph_nodes",
    "lumps in neck": "swelled_lymph_nodes",
    "lumps under jaw": "swelled_lymph_nodes",
    "lumps under arm": "swelled_lymph_nodes",
    "lumps in armpit": "swelled_lymph_nodes",
    "swollen neck": "swelled_lymph_nodes",
    "glands up": "swelled_lymph_nodes",
    "glands swollen": "swelled_lymph_nodes",
    "enlarged thyroid": "enlarged_thyroid",
    "goiter": "enlarged_thyroid",
    "goitre": "enlarged_thyroid",
    "thyroid swelling": "enlarged_thyroid",
    "thyroid big": "enlarged_thyroid",
    "thyroid issues": "enlarged_thyroid",
    "thyroid problem": "enlarged_thyroid",
    "lump in throat": "enlarged_thyroid",
    # ── nails / miscellaneous ─────────────────────────────────────────────────
    "brittle nails": "brittle_nails",
    "weak nails": "brittle_nails",
    "breaking nails": "brittle_nails",
    "peeling nails": "brittle_nails",
    "soft nails": "brittle_nails",
    "ridged nails": "brittle_nails",
    "bumpy nails": "brittle_nails",
    "inflammatory nails": "inflammatory_nails",
    "swollen nails": "inflammatory_nails",
    "infected nails": "inflammatory_nails",
    "inflamed nails": "inflammatory_nails",
    "nail fungus": "inflammatory_nails",
    "yellow nails": "inflammatory_nails",
    "discolored nails": "inflammatory_nails",
    "nail dents": "small_dents_in_nails",
    "pitted nails": "small_dents_in_nails",
    "loss of smell": "loss_of_smell",
    "cannot smell": "loss_of_smell",
    "no sense of smell": "loss_of_smell",
    "can't smell": "loss_of_smell",
    "no smell": "loss_of_smell",
    "smell gone": "loss_of_smell",
    "anosmia": "loss_of_smell",
    "lost smell": "loss_of_smell",
    "taste gone": "loss_of_smell",
    "can't taste": "loss_of_smell",
    "taste off": "loss_of_smell",
    "mouth ulcers": "ulcers_on_tongue",
    "tongue ulcers": "ulcers_on_tongue",
    "canker sores": "ulcers_on_tongue",
    "mouth sores": "ulcers_on_tongue",
    "tongue sores": "ulcers_on_tongue",
    "tongue painful": "ulcers_on_tongue",
    "white tongue": "ulcers_on_tongue",
    "coated tongue": "ulcers_on_tongue",
    "blotchy tongue": "ulcers_on_tongue",
    "toxic appearance": "toxic_look_(typhos)",
    "looks very ill": "toxic_look_(typhos)",
    "looks ill": "toxic_look_(typhos)",
    "looks sick": "toxic_look_(typhos)",
    "very ill appearance": "toxic_look_(typhos)",
    "gravely ill": "toxic_look_(typhos)",
    "irregular blood sugar": "irregular_sugar_level",
    "blood sugar fluctuations": "irregular_sugar_level",
    "high blood sugar": "irregular_sugar_level",
    "blood sugar issues": "irregular_sugar_level",
    "sugar high": "irregular_sugar_level",
    "sugar low": "irregular_sugar_level",
    "sugar crash": "irregular_sugar_level",
    "internal itching": "internal_itching",
    "itching inside": "internal_itching",
    "itching deep": "internal_itching",
    "acute liver failure": "acute_liver_failure",
    "liver failure": "acute_liver_failure",
    "liver failing": "acute_liver_failure",
    "liver shutdown": "acute_liver_failure",
    "liver problems": "acute_liver_failure",
    "liver issue": "acute_liver_failure",
    "hepatic failure": "acute_liver_failure",
    "scurring": "scurring",
    "skin scurring": "scurring",
    "pitted scars": "scurring",
}


@dataclass(frozen=True)
class ExtractionResult:
    symptoms: tuple[str, ...]
    negated_symptoms: tuple[str, ...]
    transformer_entities: tuple[str, ...]
    unmatched_terms: tuple[str, ...]
    nlp_available: bool
    nlp_note: str | None = None
    context: dict[str, str] | None = None


# Stopwords that must never trigger fuzzy symptom matching.  Including these
# would let common glue words ("and", "the", "of", "no") spuriously align
# with multi-token feature phrases that happen to share the token.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "nor", "so", "for", "yet",
    "no", "not", "without", "don", "didn", "doesn", "haven", "hasn",
    "of", "to", "in", "on", "at", "by", "with", "from", "as", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "out", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "only", "own",
    "same", "than", "too", "very", "can", "will", "just", "should",
    "now", "i", "you", "he", "she", "we", "they", "it", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their", "is", "was",
    "are", "be", "been", "being", "have", "has", "had", "having", "do",
    "does", "did", "doing", "would", "could", "may", "might", "must",
    "shall", "s", "t", "d", "m", "re", "ve", "ll", "am", "im", "ive",
    "id", "youre", "hes", "shes", "were", "theyre", "ive", "youve",
    "weve", "theyve", "youll", "itll", "well", "ill",
})


class SymptomMapper:
    """Converts NER spans and free-text phrases to approved binary features."""

    def __init__(self, feature_names: Iterable[str]) -> None:
        self.feature_names = tuple(feature_names)
        self.feature_set = set(self.feature_names)
        self.display_to_feature = {normalize_key(feature): feature for feature in self.feature_names}
        self.phrases = tuple(self.display_to_feature)
        self.aliases = {
            normalize_key(alias): self.display_to_feature[normalize_key(target)]
            for alias, target in COMMON_ALIASES.items()
            if normalize_key(target) in self.display_to_feature
        }
        # Precompute combined phrase/negation matchers. With the merged corpus
        # there are ~1,500 features; looping through every phrase and alias for
        # every user message made negation checks needlessly expensive.
        self.phrase_to_features: dict[str, set[str]] = {}
        for phrase, feature in self.display_to_feature.items():
            self.phrase_to_features.setdefault(phrase, set()).add(feature)
        for phrase, feature in self.aliases.items():
            self.phrase_to_features.setdefault(phrase, set()).add(feature)
        ordered_phrases = sorted(self.phrase_to_features, key=len, reverse=True)
        phrase_alternation = "|".join(re.escape(phrase) for phrase in ordered_phrases)
        self.exact_phrase_pattern = re.compile(
            rf"(?<![a-z0-9])(?P<phrase>{phrase_alternation})(?![a-z0-9])"
        ) if phrase_alternation else None
        cues = (
            "no", "not", "without", "don t have", "dont have", "doesn t have",
            "doesnt have", "haven t had", "havent had", "didn t have",
            "didnt have", "denies", "deny", "denied", "negative for",
            "free of", "free from", "never had", "no signs of", "no more",
            "no evidence of", "no history of", "rule out", "ruled out",
            "absent", "not experiencing", "not having", "not feeling",
            "no longer", "no longer have", "no longer experiencing",
        )
        cue_pattern = "(?:" + "|".join(re.escape(cue) for cue in cues) + ")"
        filler = r"(?:\s+(?:a|an|the|any|much|some|real|really|had|have|has|experienced|been|getting|at all|today|yesterday|recently|more|longer|this|that|these|those|notice|noticed|feel|felt|been|of|for|with|to|from)){0,5}"
        self.negation_pattern = re.compile(
            rf"\b{cue_pattern}{filler}\s+(?:(?:a|an|the|any)\s+)?(?P<phrase>{phrase_alternation})\b"
        ) if phrase_alternation else None

    def _match_phrase(self, phrase: str, fuzzy: bool = True) -> str | None:
        normalized = normalize_key(phrase)
        if not normalized:
            return None
        if normalized in self.display_to_feature:
            return self.display_to_feature[normalized]
        if normalized in self.aliases:
            return self.aliases[normalized]
        if not fuzzy or len(normalized) < 4:
            return None
        if fuzz is None or process is None:
            return None  # rapidfuzz not installed; exact/alias match only
        # Length-aware thresholds: short phrases need higher similarity to avoid
        # spurious matches; multi-word phrases can use a slightly lower bar
        # because typos and re-orderings are common.
        token_count = len(normalized.split())
        if token_count == 1:
            # Use plain ratio (not WRatio) for single tokens because WRatio's
            # partial_ratio component inflates scores when a short token like
            # "no" appears as a substring inside a longer feature name.
            threshold = 85
            scorer = fuzz.ratio
            candidate_phrases = self.phrases
        elif token_count == 2:
            threshold = 85
            scorer = fuzz.token_set_ratio
            candidate_phrases = self.phrases
        else:
            threshold = 80
            scorer = fuzz.WRatio
            # For 3+ token phrases, require at least one non-stopword token
            # overlap between query and candidate.  Without this, a phrase
            # like "and no fever" can spuriously match "cold hands and feets"
            # because they share only the stopword "and".
            content_tokens = [t for t in normalized.split() if t not in _STOPWORDS]
            if not content_tokens:
                return None
            candidate_phrases = [
                phrase for phrase in self.phrases
                if any(t in phrase.split() for t in content_tokens)
            ]
            if not candidate_phrases:
                return None
        candidate = process.extractOne(normalized, candidate_phrases, scorer=scorer)
        if candidate and candidate[1] >= threshold:
            return self.display_to_feature[candidate[0]]
        return None

    def map_text(self, text: str, entities: Iterable[str] = ()) -> tuple[set[str], set[str]]:
        """Use exact spans first, then conservative phrase-level fuzzy matching."""
        matches = set()
        normalized_text = normalize_key(text)
        unmatched: set[str] = set()

        # Compute negated features up front so the fuzzy layer can never re-add a
        # feature the negation pass already removed (e.g. "no cough" must not
        # also pull in cough via token overlap).
        negated = self.map_negated_text(text)

        # One precompiled alternation is substantially faster than compiling a
        # separate regex for each of the expanded corpus's 1,500+ symptoms.
        if self.exact_phrase_pattern is not None:
            for match in self.exact_phrase_pattern.finditer(normalized_text):
                matches.update(self.phrase_to_features.get(match.group("phrase"), ()))
        matches -= negated

        if not matches:
            for entity in entities:
                feature = self._match_phrase(entity)
                if feature and feature not in negated:
                    matches.add(feature)
                elif normalize_key(entity):
                    unmatched.add(normalize_key(entity))

        # Fuzzy matching is a fallback for wording and spelling variation.  Do
        # not run it after an exact/alias/entity match: otherwise a short phrase
        # can spuriously collect similar-looking symptoms from a large vocabulary.
        if not matches:
            tokens = normalized_text.split()
            for length in range(1, min(4, len(tokens)) + 1):
                for start in range(0, len(tokens) - length + 1):
                    phrase = " ".join(tokens[start : start + length])
                    # Skip pure-stopword phrases so tokens like "and", "the",
                    # "of" never fuzzy-match into unrelated features.
                    if all(part in _STOPWORDS for part in phrase.split()):
                        continue
                    feature = self._match_phrase(phrase)
                    if feature and feature not in negated:
                        matches.add(feature)
        return matches, unmatched

    def map_negated_text(self, text: str) -> set[str]:
        """Map symptoms after common English negation cues efficiently."""
        normalized = normalize_key(text)
        negated: set[str] = set()
        if not normalized or self.negation_pattern is None:
            return negated
        for match in self.negation_pattern.finditer(normalized):
            negated.update(self.phrase_to_features.get(match.group("phrase"), ()))
        return negated


class HuggingFaceSymptomExtractor:
    def __init__(self, mapper: SymptomMapper, model_name: str = MEDICAL_NER_MODEL) -> None:
        self.mapper = mapper
        self.model_name = model_name
        self._pipeline = None
        self._semantic_tokenizer = None
        self._semantic_model = None
        self._semantic_feature_embeddings = None
        self._load_error: str | None = None
        self._semantic_error: str | None = None

    @property
    def available(self) -> bool:
        return self._pipeline is not None or self._semantic_model is not None

    def load(self) -> None:
        """Load the public medical NER model once; download is handled by HF cache."""
        if self._pipeline is not None or self._load_error is not None:
            return
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                task="token-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                aggregation_strategy="simple",
                device=-1,
            )
        except Exception as exc:  # network/cache/device problems should not break chat
            self._load_error = f"Medical NER model unavailable ({type(exc).__name__})."

    def _load_semantic_model(self) -> None:
        """Load BioClinicalBERT only when deterministic matching found nothing."""
        if self._semantic_model is not None or self._semantic_error is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch

            self._semantic_tokenizer = AutoTokenizer.from_pretrained(MEDICAL_SEMANTIC_MODEL)
            self._semantic_model = AutoModel.from_pretrained(MEDICAL_SEMANTIC_MODEL)
            self._semantic_model.eval()
            with torch.no_grad():
                self._semantic_feature_embeddings = self._encode_texts(list(self.mapper.feature_names), torch)
        except Exception as exc:
            self._semantic_error = f"Clinical semantic model unavailable ({type(exc).__name__})."
            self._semantic_tokenizer = None
            self._semantic_model = None

    def _encode_texts(self, texts: list[str], torch_module):
        encoded = self._semantic_tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        with torch_module.no_grad():
            output = self._semantic_model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(output.size()).float()
        pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return torch_module.nn.functional.normalize(pooled, p=2, dim=1)

    def _semantic_matches(self, text: str) -> set[str]:
        self._load_semantic_model()
        if self._semantic_model is None or self._semantic_feature_embeddings is None:
            return set()
        try:
            import torch
            query = self._encode_texts([text], torch)
            scores = torch.matmul(query, self._semantic_feature_embeddings.T)[0]
            top_k = max(1, int(MEDICAL_SEMANTIC_TOP_K))
            ranked = torch.argsort(scores, descending=True)[:top_k].tolist()
            chosen: list[str] = []
            for index in ranked:
                score = float(scores[index].item())
                if score < MEDICAL_SEMANTIC_THRESHOLD:
                    break
                chosen.append(self.mapper.feature_names[index])
            return set(chosen)
        except Exception as exc:
            self._semantic_error = f"Clinical semantic inference unavailable ({type(exc).__name__})."
            return set()

    def extract(self, text: str) -> ExtractionResult:
        """Run Hugging Face NER on every message, then map spans to ML features."""
        self.load()
        entities: list[str] = []
        if self._pipeline is not None:
            try:
                raw_entities = self._pipeline(text)
                entities = [
                    str(entity.get("word", "")).replace("##", "").strip()
                    for entity in raw_entities
                    if float(entity.get("score", 0.0)) >= 0.35 and str(entity.get("word", "")).strip()
                ]
            except Exception as exc:
                self._load_error = f"Medical NER inference unavailable ({type(exc).__name__})."
                self._pipeline = None

        symptoms, unmatched = self.mapper.map_text(text, entities)
        # Semantic matching is deliberately a last-resort, one-symptom fallback.
        # Exact phrases, aliases, fuzzy matches, and NER must take priority.
        if not symptoms:
            symptoms.update(self._semantic_matches(text))
        negated = self.mapper.map_negated_text(text)
        symptoms -= negated
        context = self.extract_context(text)
        return ExtractionResult(
            symptoms=tuple(sorted(symptoms)),
            negated_symptoms=tuple(sorted(negated)),
            transformer_entities=tuple(entities),
            unmatched_terms=tuple(sorted(unmatched)),
            nlp_available=self._pipeline is not None,
            nlp_note=self._load_error or self._semantic_error,
            context=context,
        )

    @staticmethod
    def extract_context(text: str) -> dict[str, str]:
        """Extract lightweight, explainable context signals from the raw message."""
        value = text.lower()
        duration = None
        match = re.search(r"(?:for|since)\s+(\d+\s+(?:hours?|days?|weeks?|months?))", value)
        if match:
            duration = match.group(1)
        severity = next((word for word in ("mild", "moderate", "severe", "very severe") if word in value), None)
        worsening = "worsening" if any(phrase in value for phrase in ("getting worse", "worsening", "worse today", "increasing")) else None
        age_group = None
        age_match = re.search(r"\b(\d{1,3})\s*(?:years? old|year-old|yo)\b", value)
        if age_match:
            age = int(age_match.group(1))
            age_group = "child" if age < 18 else "older adult" if age >= 65 else "adult"
        else:
            group_match = re.search(r"\bage group\s+(child|adult|older adult)\b", value)
            if group_match:
                age_group = group_match.group(1)
        return {key: value for key, value in {"duration": duration, "severity": severity, "worsening": worsening, "age_group": age_group}.items() if value}
