import { useState, useEffect } from 'react'
import axios from 'axios'
import { supabase } from './supabaseClient'
import Auth from './Auth'

const API_BASE = "http://127.0.0.1:5000"

// -----------------------------------------------------------------------
// FIELD CATALOG — the single source of truth for the entire form.
//
// Each field is asked ONCE, grouped by medical category (not by disease).
// "targets" describes how that one answer gets written into each disease's
// prediction payload, including any unit/format conversion needed.
//
// NOTE on ckd.Gender: assumed convention is Female -> 0, Male -> 1
// (sklearn LabelEncoder's default alphabetical mapping). Verify this
// against your own training notebook before trusting CKD gender-based
// predictions.
// -----------------------------------------------------------------------

function mapGenderToHeart(gender) {
  // The Heart Failure dataset only contains Male/Female. "Other" has no
  // equivalent in that training data, so it falls back to Female.
  return gender === "Male" ? "M" : "F"
}

function mapGenderToCKD(gender) {
  // ASSUMED convention — verify against your notebook.
  if (gender === "Male") return "1"
  return "0"
}

function mapSmokingToCKD(smokingHistory) {
  // The CKD dataset only has a binary current-smoker flag.
  // Only an active "current" smoker maps to Yes.
  return smokingHistory === "current" ? "1" : "0"
}

const FIELD_CATALOG = [
  {
    category: "Personal Information",
    fields: [
      {
        id: "fullName",
        label: "Full Name",
        description: "Used to label your report. Not used in any risk calculation.",
        type: "text",
        required: true,
        targets: []
      },

      {
        id: "age",
        label: "Age",
        description: "Your age in years.",
        type: "number",
        required: true,
        min: 1,
        max: 120,
        targets: [
          { disease: "diabetes", field: "age" },
          { disease: "heart", field: "Age" },
          { disease: "ckd", field: "Age" }
        ]
      },

      {
        id: "gender",
        label: "Gender",
        description: "Used because some risk factors differ by sex.",
        type: "select",
        required: true,
        options: ["Male", "Female", "Other"],
        targets: [
          {
            disease: "diabetes",
            field: "gender",
            transform: v => v
          },
          {
            disease: "heart",
            field: "Sex",
            transform: mapGenderToHeart
          },
          {
            disease: "ckd",
            field: "Gender",
            transform: mapGenderToCKD
          }
        ]
      },

      {
        id: "smokingHistory",
        label: "Smoking History",
        description: "Your current and past smoking habits.",
        type: "select",
        required: true,
        options: [
          "never",
          "current",
          "former",
          "ever",
          "not current",
          "No Info"
        ],
        optionLabels: {
          never: "Never smoked",
          current: "Currently smoke",
          former: "Former smoker",
          ever: "Smoked at some point",
          "not current": "Used to smoke, not currently",
          "No Info": "Prefer not to say"
        },
        targets: [
          {
            disease: "diabetes",
            field: "smoking_history",
            transform: v => v
          },
          {
            disease: "ckd",
            field: "Smoking",
            transform: mapSmokingToCKD
          }
        ]
      }
    ]
  },

  {
    category: "Body Measurements",
    fields: [
      {
        id: "bmi",
        label: "BMI (Body Mass Index)",
        description:
          "Weight(kg) divided by height(m) squared. Ask your clinic or use an online BMI calculator if you don't have this.",
        type: "number",
        required: true,
        step: "0.1",
        min: 10,
        max: 70,
        targets: [
          {
            disease: "diabetes",
            field: "bmi"
          },
          {
            disease: "ckd",
            field: "BMI"
          }
        ]
      }
    ]
  },

  {
    category: "Blood Pressure",
    fields: [
      {
        id: "systolicBP",
        label: "Systolic Blood Pressure",
        description:
          "The higher number in a blood pressure reading, e.g. the 120 in 120/80.",
        type: "number",
        required: true,
        min: 60,
        max: 250,
        targets: [
          {
            disease: "heart",
            field: "RestingBP"
          },
          {
            disease: "ckd",
            field: "SystolicBP"
          }
        ]
      },

      {
        id: "diastolicBP",
        label: "Diastolic Blood Pressure",
        description:
          "The lower number in a blood pressure reading, e.g. the 80 in 120/80. Optional — leave blank if unknown.",
        type: "number",
        required: false,
        min: 40,
        max: 150,
        targets: [
          {
            disease: "ckd",
            field: "DiastolicBP"
          }
        ]
      }
    ]
  },

  {
    category: "Blood Sugar",
    fields: [
      {
        id: "bloodGlucose",
        label: "Blood Glucose Level (random reading)",
        description:
          "A blood sugar reading taken at any time of day, not necessarily fasting. In mg/dL.",
        type: "number",
        required: true,
        min: 40,
        max: 400,
        targets: [
          {
            disease: "diabetes",
            field: "blood_glucose_level"
          }
        ]
      },

      {
        id: "fastingBloodSugar",
        label: "Fasting Blood Sugar",
        description:
          "A blood sugar reading taken after at least 8 hours without food. In mg/dL.",
        type: "number",
        required: true,
        min: 40,
        max: 400,
        targets: [
          // Heart model only needs a Yes/No flag for ">120"
          {
            disease: "heart",
            field: "FastingBS",
            transform: v =>
              parseFloat(v) > 120 ? "1" : "0"
          },

          {
            disease: "ckd",
            field: "FastingBloodSugar"
          }
        ]
      },

      {
        id: "hba1c",
        label: "HbA1c Level",
        description:
          "Average blood sugar over the past 2-3 months, from a standard HbA1c blood test (%).",
        type: "number",
        required: true,
        step: "0.1",
        min: 3,
        max: 15,
        targets: [
          {
            disease: "diabetes",
            field: "HbA1c_level"
          },
          {
            disease: "ckd",
            field: "HbA1c"
          }
        ]
      }
    ]
  },

  {
    category: "Cholesterol",
    fields: [
      {
        id: "totalCholesterol",
        label: "Total Cholesterol",
        description:
          "From a standard lipid panel, in mg/dL.",
        type: "number",
        required: true,
        min: 80,
        max: 500,
        targets: [
          {
            disease: "heart",
            field: "Cholesterol"
          },
          {
            disease: "ckd",
            field: "CholesterolTotal"
          }
        ]
      },

      {
        id: "ldl",
        label: "LDL Cholesterol",
        description:
          '"Bad" cholesterol, from the same lipid panel. Optional.',
        type: "number",
        required: false,
        min: 20,
        max: 300,
        targets: [
          {
            disease: "ckd",
            field: "CholesterolLDL"
          }
        ]
      },

      {
        id: "hdl",
        label: "HDL Cholesterol",
        description:
          '"Good" cholesterol, from the same lipid panel. Optional.',
        type: "number",
        required: false,
        min: 10,
        max: 150,
        targets: [
          {
            disease: "ckd",
            field: "CholesterolHDL"
          }
        ]
      },

      {
        id: "triglycerides",
        label: "Triglycerides",
        description:
          "From the same lipid panel. Optional.",
        type: "number",
        required: false,
        min: 30,
        max: 600,
        targets: [
          {
            disease: "ckd",
            field: "CholesterolTriglycerides"
          }
        ]
      }
    ]
  },

  // -----------------------------------------------------------------------
  // KIDNEY FUNCTION LABS
  //
  // Sodium and Potassium are REQUIRED because the CKD backend currently
  // requires:
  //   SerumElectrolytesSodium
  //   SerumElectrolytesPotassium
  // -----------------------------------------------------------------------

  {
    category: "Kidney Function Labs",
    description:
      "These come from a more detailed kidney panel. Sodium and Potassium are required for the CKD prediction. Other kidney values can be left blank if unavailable.",
    fields: [
      {
        id: "creatinine",
        label: "Serum Creatinine",
        description:
          "Key kidney function marker, mg/dL.",
        type: "number",
        required: false,
        step: "0.1",
        min: 0.2,
        max: 15,
        targets: [
          {
            disease: "ckd",
            field: "SerumCreatinine"
          }
        ]
      },

      {
        id: "bun",
        label: "Blood Urea Nitrogen (BUN)",
        description:
          "Kidney function marker, mg/dL.",
        type: "number",
        required: false,
        min: 2,
        max: 150,
        targets: [
          {
            disease: "ckd",
            field: "BUNLevels"
          }
        ]
      },

      {
        id: "gfr",
        label: "GFR (Glomerular Filtration Rate)",
        description:
          "Estimates how well your kidneys filter blood, mL/min.",
        type: "number",
        required: false,
        min: 1,
        max: 150,
        targets: [
          {
            disease: "ckd",
            field: "GFR"
          }
        ]
      },

      {
        id: "proteinInUrine",
        label: "Protein in Urine",
        description:
          "From a urinalysis, g/day.",
        type: "number",
        required: false,
        step: "0.1",
        min: 0,
        max: 15,
        targets: [
          {
            disease: "ckd",
            field: "ProteinInUrine"
          }
        ]
      },

      {
        id: "acr",
        label: "Albumin-Creatinine Ratio (ACR)",
        description:
          "From a urine test, mg/g.",
        type: "number",
        required: false,
        min: 0,
        max: 1000,
        targets: [
          {
            disease: "ckd",
            field: "ACR"
          }
        ]
      },

      {
        id: "sodium",
        label: "Sodium",
        description:
          "Blood electrolyte level, mEq/L. Required for CKD prediction.",
        type: "number",
        required: true,
        min: 110,
        max: 160,
        targets: [
          {
            disease: "ckd",
            field: "SerumElectrolytesSodium"
          }
        ]
      },

      {
        id: "potassium",
        label: "Potassium",
        description:
          "Blood electrolyte level, mEq/L. Required for CKD prediction.",
        type: "number",
        required: true,
        step: "0.1",
        min: 2,
        max: 8,
        targets: [
          {
            disease: "ckd",
            field: "SerumElectrolytesPotassium"
          }
        ]
      },

      {
        id: "calcium",
        label: "Calcium",
        description:
          "Blood electrolyte level, mg/dL.",
        type: "number",
        required: false,
        step: "0.1",
        min: 5,
        max: 14,
        targets: [
          {
            disease: "ckd",
            field: "SerumElectrolytesCalcium"
          }
        ]
      },

      {
        id: "phosphorus",
        label: "Phosphorus",
        description:
          "Blood electrolyte level, mg/dL.",
        type: "number",
        required: false,
        step: "0.1",
        min: 1,
        max: 10,
        targets: [
          {
            disease: "ckd",
            field: "SerumElectrolytesPhosphorus"
          }
        ]
      },

      {
        id: "hemoglobin",
        label: "Hemoglobin",
        description:
          "From a complete blood count, g/dL.",
        type: "number",
        required: false,
        step: "0.1",
        min: 3,
        max: 20,
        targets: [
          {
            disease: "ckd",
            field: "HemoglobinLevels"
          }
        ]
      }
    ]
  },

  {
    category: "Symptoms",
    fields: [
      {
        id: "chestPainType",
        label: "Chest Pain Type",
        description:
          "The kind of chest discomfort you experience, if any.",
        type: "select",
        required: true,
        options: [
          "ATA",
          "NAP",
          "ASY",
          "TA"
        ],
        optionLabels: {
          ATA: "Comes with exertion, eases with rest (typical angina pattern)",
          NAP: "Chest pain not clearly linked to exertion",
          ASY: "No chest pain",
          TA: "Brief, sharp chest pain"
        },
        targets: [
          {
            disease: "heart",
            field: "ChestPainType"
          }
        ]
      },

      {
        id: "exerciseAngina",
        label: "Chest Pain During Exercise",
        description:
          "Do you get chest pain or tightness during physical activity?",
        type: "select",
        required: true,
        options: [
          "N",
          "Y"
        ],
        optionLabels: {
          N: "No",
          Y: "Yes"
        },
        targets: [
          {
            disease: "heart",
            field: "ExerciseAngina"
          }
        ]
      },

      {
        id: "edema",
        label: "Swelling in Legs, Ankles or Feet (Edema)",
        description:
          "Optional.",
        type: "select",
        required: false,
        options: [
          "0",
          "1"
        ],
        optionLabels: {
          "0": "No",
          "1": "Yes"
        },
        targets: [
          {
            disease: "ckd",
            field: "Edema"
          }
        ]
      },

      {
        id: "fatigueLevel",
        label: "Fatigue Level",
        description:
          "0 = none, 10 = severe. Optional, defaults to 0 if left blank.",
        type: "number",
        required: false,
        min: 0,
        max: 10,
        defaultValue: "0",
        targets: [
          {
            disease: "ckd",
            field: "FatigueLevels"
          }
        ]
      },

      {
        id: "nauseaLevel",
        label: "Nausea / Vomiting Level",
        description:
          "0 = none, 10 = severe. Optional, defaults to 0 if left blank.",
        type: "number",
        required: false,
        min: 0,
        max: 10,
        defaultValue: "0",
        targets: [
          {
            disease: "ckd",
            field: "NauseaVomiting"
          }
        ]
      },

      {
        id: "muscleCrampsLevel",
        label: "Muscle Cramps Level",
        description:
          "0 = none, 10 = severe. Optional, defaults to 0 if left blank.",
        type: "number",
        required: false,
        min: 0,
        max: 10,
        defaultValue: "0",
        targets: [
          {
            disease: "ckd",
            field: "MuscleCramps"
          }
        ]
      },

      {
        id: "itchingLevel",
        label: "Itching Level",
        description:
          "0 = none, 10 = severe. Optional, defaults to 0 if left blank.",
        type: "number",
        required: false,
        min: 0,
        max: 10,
        defaultValue: "0",
        targets: [
          {
            disease: "ckd",
            field: "Itching"
          }
        ]
      }
    ]
  },

  {
    category: "Medical History",
    fields: [
      {
        id: "hasHypertension",
        label: "Do you personally have hypertension (high blood pressure)?",
        description:
          "Your own diagnosis, not a family member's.",
        type: "select",
        required: true,
        options: [
          "0",
          "1"
        ],
        optionLabels: {
          "0": "No",
          "1": "Yes"
        },
        targets: [
          {
            disease: "diabetes",
            field: "hypertension"
          }
        ]
      },

      {
        id: "hasHeartDisease",
        label: "Do you have a personal history of heart disease?",
        description: "",
        type: "select",
        required: true,
        options: [
          "0",
          "1"
        ],
        optionLabels: {
          "0": "No",
          "1": "Yes"
        },
        targets: [
          {
            disease: "diabetes",
            field: "heart_disease"
          }
        ]
      },

      {
        id: "previousAKI",
        label: "Have you had a previous acute kidney injury?",
        description:
          "Optional.",
        type: "select",
        required: false,
        options: [
          "0",
          "1"
        ],
        optionLabels: {
          "0": "No",
          "1": "Yes"
        },
        targets: [
          {
            disease: "ckd",
            field: "PreviousAcuteKidneyInjury"
          }
        ]
      },

      {
        id: "utiHistory",
        label: "History of urinary tract infections?",
        description:
          "Optional.",
        type: "select",
        required: false,
        options: [
          "0",
          "1"
        ],
        optionLabels: {
          "0": "No",
          "1": "Yes"
        },
        targets: [
          {
            disease: "ckd",
            field: "UrinaryTractInfections"
          }
        ]
      },

      {
        id: "familyKidneyDisease",
        label: "Family history of kidney disease?",
        description:
          "Optional.",
        type: "select",
        required: false,
        options: [
          "0",
          "1"
        ],
        optionLabels: {
          "0": "No",
          "1": "Yes"
        },
        targets: [
          {
            disease: "ckd",
            field: "FamilyHistoryKidneyDisease"
          }
        ]
      },

      {
        id: "familyHypertension",
        label: "Family history of hypertension?",
        description:
          "Optional.",
        type: "select",
        required: false,
        options: [
          "0",
          "1"
        ],
        optionLabels: {
          "0": "No",
          "1": "Yes"
        },
        targets: [
          {
            disease: "ckd",
            field: "FamilyHistoryHypertension"
          }
        ]
      },

      {
        id: "familyDiabetes",
        label: "Family history of diabetes?",
        description:
          "Optional.",
        type: "select",
        required: false,
        options: [
          "0",
          "1"
        ],
        optionLabels: {
          "0": "No",
          "1": "Yes"
        },
        targets: [
          {
            disease: "ckd",
            field: "FamilyHistoryDiabetes"
          }
        ]
      }
    ]
  }
]


// Placeholder hints


const PLACEHOLDER_HINTS = {
  fullName: "e.g. Ahmed Khan",
  age: "e.g. 45"
}

function getPlaceholder(field) {
  if (PLACEHOLDER_HINTS[field.id]) {
    return PLACEHOLDER_HINTS[field.id]
  }

  if (field.min !== undefined && field.max !== undefined) {
    return `Range: ${field.min}-${field.max}`
  }

  return ""
}


// Flattened lookup used for building the submission payload


const ALL_FIELDS = FIELD_CATALOG.flatMap(
  group => group.fields
)


// Build prediction payload


function buildPayload(values) {
  const payload = {
    diabetes: {},
    heart: {},
    ckd: {}
  }

  for (const field of ALL_FIELDS) {
    const raw = values[field.id]

    const value =
      raw === undefined || raw === ""
        ? field.defaultValue
        : raw

    // Truly unanswered optional field
    if (value === undefined || value === "") {
      continue
    }

    for (const target of field.targets) {
      const finalValue = target.transform
        ? target.transform(value)
        : value

      payload[target.disease][target.field] =
        field.type === "number"
          ? parseFloat(finalValue)
          : finalValue
    }
  }

  return payload
}


// Validate required fields


function missingRequiredFields(values) {
  return ALL_FIELDS.filter(
    field =>
      field.required &&
      (
        values[field.id] === undefined ||
        values[field.id] === ""
      )
  )
}


// Renders simple Markdown (## headers, * bullets, plain paragraphs) as
// HTML, without pulling in a full markdown library. Used for the
// Gemini-generated ai_explanation text.

function MarkdownLite({ text }) {
  const lines = text.split("\n").filter(l => l.trim() !== "")

  return (
    <div className="text-sm space-y-1">
      {lines.map((line, i) => {
        const trimmed = line.trim()
        if (trimmed.startsWith("## ")) {
          return <p key={i} className="font-semibold mt-2">{trimmed.slice(3)}</p>
        }
        if (trimmed.startsWith("* ") || trimmed.startsWith("- ")) {
          return <p key={i} className="ml-3">• {trimmed.slice(2).replace(/\*\*/g, "")}</p>
        }
        return <p key={i}>{trimmed.replace(/\*\*/g, "")}</p>
      })}
    </div>
  )
}

// Main App

function App() {
  const [session, setSession] = useState(null)
  const [checkingSession, setCheckingSession] = useState(true)
  const [backendUp, setBackendUp] = useState(null)
  const [values, setValues] = useState({})
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [extracting, setExtracting] = useState(false)
  const [extractMessage, setExtractMessage] = useState(null)

  // ---------------------------------------------------------------------
  // Check backend
  // ---------------------------------------------------------------------

  useEffect(() => {
    axios
      .get(`${API_BASE}/health`)
      .then(() => setBackendUp(true))
      .catch(() => setBackendUp(false))
  }, [])

  useEffect(() => {
    // On load, check if a session already exists (e.g. page was refreshed
    // while logged in).
    supabase.auth.getSession().then(({ data }) => {
      setSession(data.session)
      setCheckingSession(false)
    })

    // Keep session state in sync with login/logout/token refresh events.
    const { data: authListener } = supabase.auth.onAuthStateChange((_event, newSession) => {
      setSession(newSession)
    })

    return () => {
      authListener.subscription.unsubscribe()
    }
  }, [])

  // ---------------------------------------------------------------------
  // Handle field change
  // ---------------------------------------------------------------------

  const handleSignOut = async () => {
    await supabase.auth.signOut()
  }

  const handleChange = (fieldId, value) => {
    setValues(prev => ({
      ...prev,
      [fieldId]: value
    }))
  }

  // ---------------------------------------------------------------------
  // Handle blood report PDF/image upload -> auto-fill fields
  // ---------------------------------------------------------------------

  const handleFileUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return

    setExtracting(true)
    setExtractMessage(null)

    const formData = new FormData()
    formData.append("report", file)

    try {
      const res = await axios.post(`${API_BASE}/extract-report`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
          Authorization: `Bearer ${session.access_token}`
        }
      })

      const extracted = res.data.extracted_fields

      if (!extracted || Object.keys(extracted).length === 0) {
        setExtractMessage("Report se koi field nahi mil saki. Manually fill kar lo.")
      } else {
        setValues(prev => ({ ...prev, ...extracted }))
        setExtractMessage(
          `${Object.keys(extracted).length} fields report se bhar diye gaye. Please neeche review karo.`
        )
      }
    } catch (err) {
      console.error("Extraction error:", err)
      setExtractMessage("Extraction fail hui. Manually fill karo.")
    } finally {
      setExtracting(false)
    }
  }

  // ---------------------------------------------------------------------
  // Submit prediction request
  // ---------------------------------------------------------------------

  const handleSubmit = async () => {
    setError(null)
    setResult(null)

    // Validate required fields
    const missing = missingRequiredFields(values)

    if (missing.length > 0) {
      setError(
        `Please fill in: ${missing
          .map(field => field.label)
          .join(", ")}`
      )
      return
    }

    setLoading(true)

    try {
      const payload = buildPayload(values)

      console.log("Prediction payload:", payload)

      // /report/full runs the ML prediction AND generates the RAG+Gemini
      // explanation/recommendations for each disease in one call.
      const res = await axios.post(
        `${API_BASE}/report/full`,
        payload,
        {
          headers: { Authorization: `Bearer ${session.access_token}` }
        }
      )

      setResult(res.data)
    } catch (err) {
      console.error("Prediction error:", err)

      setError(
        err.response?.data?.error ||
        "Something went wrong. Check the browser console for details."
      )
    } finally {
      setLoading(false)
    }
  }

  // ---------------------------------------------------------------------
  // Backend unavailable
  // ---------------------------------------------------------------------

  if (checkingSession) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-stone-50">
        <p className="text-slate-400 text-sm">Loading...</p>
      </div>
    )
  }

  if (!session) {
    return <Auth onLoginSuccess={(newSession) => setSession(newSession)} />
  }

  if (backendUp === false) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-stone-50">
        <p className="text-slate-600">
          Can't reach the server. Is the Flask backend running?
        </p>
      </div>
    )
  }

  // ---------------------------------------------------------------------
  // Main UI
  // ---------------------------------------------------------------------

  return (
    <div className="min-h-screen bg-stone-50">

      <header className="border-b border-stone-200 bg-white">
        <div className="max-w-2xl mx-auto px-6 py-8">

          <div className="flex items-center gap-3 mb-1 justify-between">
            <h1 className="text-2xl font-semibold text-slate-800">
              Health Risk Assessment
            </h1>
            <button
              onClick={handleSignOut}
              className="ml-auto text-sm text-slate-500 hover:text-slate-700"
            >
              Sign out
            </button>
          </div>

          <p className="text-slate-500 mt-1 text-sm">
            Answer what you can from a recent blood report or your own
            knowledge. Fields marked with a dot are required; everything
            else is optional.
          </p>

        </div>
      </header>

      <main className="max-w-2xl mx-auto px-6 py-8">

        <div className="bg-teal-50 border border-teal-200 rounded-md p-6 mb-8">
          <h2 className="text-base font-semibold text-slate-800 mb-1">
            Option: Apni Blood Report Upload Karo
          </h2>
          <p className="text-sm text-slate-500 mb-3">
            Upload your lab reports one at a time — CBC, lipid panel, HbA1c,
            or kidney panel. Each upload adds to your existing data without
            erasing what's already filled in
          </p>
          <input
            type="file"
            accept=".pdf,image/*"
            onChange={handleFileUpload}
            disabled={extracting}
            className="text-sm"
          />
          {extracting && (
            <p className="text-sm text-teal-700 mt-2">Report padh rahe hain...</p>
          )}
          {extractMessage && (
            <p className="text-sm text-slate-700 mt-2">{extractMessage}</p>
          )}
        </div>

        {FIELD_CATALOG.map(group => (
          <CategorySection
            key={group.category}
            group={group}
            values={values}
            onChange={handleChange}
          />
        ))}

        <button
          onClick={handleSubmit}
          disabled={loading}
          className="w-full bg-teal-700 text-white font-medium py-3 rounded-md hover:bg-teal-800 disabled:bg-stone-300 transition-colors"
        >
          {loading
            ? "Calculating..."
            : "Get My Risk Report"}
        </button>

        {error && (
          <p className="text-red-600 mt-4 text-sm">
            {error}
          </p>
        )}

        {result && (
          <ResultReport
            result={result}
            name={values.fullName}
          />
        )}

      </main>
    </div>
  )
}

// -----------------------------------------------------------------------
// Category Section
// -----------------------------------------------------------------------

function CategorySection({
  group,
  values,
  onChange
}) {
  return (
    <section className="mb-8 pb-8 border-b border-stone-200 last:border-b-0">

      <h2 className="text-base font-semibold text-slate-800 mb-1">
        {group.category}
      </h2>

      {group.description && (
        <p className="text-sm text-slate-500 mb-4">
          {group.description}
        </p>
      )}

      <div className="space-y-4">

        {group.fields.map(field => (
          <FieldInput
            key={field.id}
            field={field}
            value={values[field.id] || ""}
            onChange={value =>
              onChange(field.id, value)
            }
          />
        ))}

      </div>
    </section>
  )
}

// -----------------------------------------------------------------------
// Field Input
// -----------------------------------------------------------------------

function FieldInput({
  field,
  value,
  onChange
}) {
  return (
    <div>

      <label className="flex items-baseline gap-1.5 text-sm font-medium text-slate-700 mb-1">

        {field.label}

        {field.required && (
          <span
            className="text-teal-600 text-xs"
            aria-hidden="true"
          >
            ●
          </span>
        )}

      </label>

      {field.description && (
        <p className="text-xs text-slate-500 mb-1.5">
          {field.description}
        </p>
      )}

      {field.type === "select" ? (

        <select
          className="w-full border border-stone-300 rounded-md px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-teal-600"
          value={value}
          onChange={e =>
            onChange(e.target.value)
          }
        >

          <option value="">
            Select...
          </option>

          {field.options.map(opt => (
            <option
              key={opt}
              value={opt}
            >
              {field.optionLabels?.[opt] || opt}
            </option>
          ))}

        </select>

      ) : (

        <input
          type={field.type}
          step={field.step}
          min={field.min}
          max={field.max}
          placeholder={getPlaceholder(field)}
          className="w-full border border-stone-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-teal-600"
          value={value}
          onChange={e =>
            onChange(e.target.value)
          }
        />

      )}

    </div>
  )
}

// -----------------------------------------------------------------------
// Result Report
// -----------------------------------------------------------------------

function ResultReport({
  result,
  name
}) {
  const categoryStyle = {
    Low: "bg-emerald-50 border-emerald-300 text-emerald-800",
    Medium: "bg-amber-50 border-amber-300 text-amber-800",
    High: "bg-rose-50 border-rose-300 text-rose-800"
  }

  // Safety check in case API returns an error structure
  // without a complete summary/results object.
  if (!result || !result.results) {
    return (
      <div className="mt-10">

        <div className="bg-red-50 border border-red-300 text-red-800 rounded-md p-4">

          <p className="font-medium">
            Prediction could not be completed.
          </p>

          {result?.error && (
            <p className="text-sm mt-1">
              {result.error}
            </p>
          )}

        </div>

      </div>
    )
  }

  return (
    <div className="mt-10">

      <h2 className="text-lg font-semibold text-slate-800 mb-4">
        {name
          ? `${name}'s Risk Report`
          : "Risk Report"}
      </h2>

      {result.summary && (
        <div className="bg-slate-800 text-white rounded-md p-4 mb-4">

          <p className="text-sm text-slate-300">
            Highest Risk
          </p>

          <p className="font-medium">
            {result.summary.highest_risk_disease}
            {" — "}
            {result.summary.highest_risk_category}
          </p>

        </div>
      )}

      {Object.values(result.results).map(r => (

        <div
          key={r.disease}
          className={`border rounded-md p-4 mb-3 ${
            categoryStyle[r.risk_category] ||
            "bg-white border-stone-300 text-slate-800"
          }`}
        >

          <p className="font-medium">
            {r.disease}
          </p>

          <p className="text-sm mt-1">
            Probability:{" "}
            {(r.probability * 100).toFixed(1)}%
          </p>

          <p className="text-sm">
            Risk Category:{" "}
            {r.risk_category}
          </p>

          {r.disclaimer && (
            <p className="text-xs mt-2 italic">
              {r.disclaimer}
            </p>
          )}

          {r.warnings &&
            r.warnings.length > 0 && (
              <div className="mt-2">

                {r.warnings.map(
                  (warning, index) => (
                    <p
                      key={index}
                      className="text-xs"
                    >
                      ⚠ {warning}
                    </p>
                  )
                )}

              </div>
            )}

                    {r.explanation?.top_factors?.length > 0 && (
            <div className="mt-3 pt-3 border-t border-current/20">
              <p className="text-xs font-medium mb-2">Top Contributing Factors</p>
              <div className="space-y-1.5">
                {(() => {
                  const maxImpact = Math.max(...r.explanation.top_factors.map(f => f.impact))
                  return r.explanation.top_factors.map((factor, i) => (
                    <div key={i} className="flex items-center gap-2 text-xs">
                      <span className={factor.direction === "increased" ? "text-rose-600" : "text-emerald-600"}>
                        {factor.direction === "increased" ? "▲" : "▼"}
                      </span>
                      <span className="w-36 truncate flex-shrink-0">{factor.feature}</span>
                      <div className="flex-1 h-1.5 bg-black/10 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${factor.direction === "increased" ? "bg-rose-500" : "bg-emerald-500"}`}
                          style={{ width: `${(factor.impact / maxImpact) * 100}%` }}
                        />
                      </div>
                    </div>
                  ))
                })()}
              </div>
            </div>
          )}

          {r.ai_explanation && (
            <div className="mt-3 pt-3 border-t border-current/20">
              <MarkdownLite text={r.ai_explanation} />
              {r.evidence_sources && r.evidence_sources.length > 0 && (
                <p className="text-xs mt-2 opacity-70">
                  Source: {r.evidence_sources.join(", ")}
                </p>
              )}
            </div>
          )}

        </div>

      ))}

      <p className="text-xs text-slate-500 mt-4">
        This is not a medical diagnosis. Please consult a doctor for any
        health concerns.
      </p>

    </div>
  )
}

export default App