### Part 4 — LLM-Powered Feature: Structured Extraction, Tabular Batch Scoring, or Model Prediction Explanation

## FEATURE track selection
Selected Track: (B) Tabular Record Batch Scoring

This track closely aligns with the objective of this project. The goal of the AI Placement Predictor is not only to predict whether a student is likely to be placed but also to assess the student's overall placement readiness and identify areas that need improvement.

In this implementation, three student records from the cleaned dataset are passed to the LLM individually. Using a predefined business rubric, the LLM evaluates each student's readiness and returns a structured JSON assessment containing fields such as the risk tier, review flag, primary readiness signal, confidence level, and a recommended action.

This approach closely matches the intended real-world use case of the project, where students can receive a readiness assessment along with actionable recommendations rather than just a prediction. Therefore, Feature Track B was selected as it best supports the project's objective of evaluating placement readiness using structured LLM-based scoring.

## TASK 1
- The api key is stored and loaded into the file.
- Validated the api call is working properly, and got the result as hello.

## TASK 2
- Prompt designing for the LLM is completed

## TASK 3
**Validation outcome for 3 students**
| Input Record                                                                                                                                                                                                                         | LLM Assessment JSON                                                                                                                                                                                                                        | Validation Status |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------- |
| `{"CGPA":77.0,"Internships":10.0,"Projects":20.0,"Workshops/Certifications":10.0,"AptitudeTestScore":800.0,"SoftSkillsRating":44.0,"ExtracurricularActivities":"yes","PlacementTraining":"yes","SSC_Marks":700.0,"HSC_Marks":730.0}` | `{"risk_tier":"medium","flag_for_review":true,"primary_signal":"AptitudeTestScore of 800 is outside the expected scale","confidence":"low","recommended_action":"Verify and normalize all numeric fields before reassessment"}`            | **Pass**          |
| `{"CGPA":7.6,"Internships":1.0,"Projects":1.0,"Workshops/Certifications":1.0,"AptitudeTestScore":71.0,"SoftSkillsRating":4.2,"ExtracurricularActivities":"no","PlacementTraining":"no","SSC_Marks":64.0,"HSC_Marks":71.0}`           | `{"risk_tier":"medium","flag_for_review":false,"primary_signal":"PlacementTraining is absent","confidence":"high","recommended_action":"Complete placement training with targeted aptitude practice"}`                                     | **Pass**          |
| `{"CGPA":7.4,"Internships":2.0,"Projects":3.0,"Workshops/Certifications":3.0,"AptitudeTestScore":86.0,"SoftSkillsRating":4.2,"ExtracurricularActivities":"yes","PlacementTraining":"yes","SSC_Marks":60.0,"HSC_Marks":75.0}`         | `{"risk_tier":"medium","flag_for_review":false,"primary_signal":"CGPA is below the strong-readiness benchmark of 8.0","confidence":"high","recommended_action":"Strengthen academic performance while maintaining placement preparation"}` | **Pass**          |


## TASK 4
The original dataset contained a `StudentID` column, but it was removed during preprocessing because it is an identifier rather than a predictive feature. As a result, the records sent to the LLM contain only the features relevant for placement-readiness assessment. In addition, a PII guardrail was implemented to detect email addresses and phone numbers before every LLM call. If either is detected, the request is blocked and not sent to the LLM.


## TASK 5

| Input                                                                                                                                                                                                                                                   | LLM Output                                                                                                                                                                                                                                                    | Valid JSON (Pass/Fail) | Pass/Block (Guardrail Result) |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- | ----------------------------- |
| `{"CGPA": 77.0, "Internships": 10.0, "Projects": 20.0, "Workshops/Certifications": 10.0, "AptitudeTestScore": 800.0, "SoftSkillsRating": 44.0, "ExtracurricularActivities": "yes", "PlacementTraining": "yes", "SSC_Marks": 700.0, "HSC_Marks": 730.0}` | `{"risk_tier":"low","flag_for_review":false,"primary_signal":"CGPA is at least 8.0","confidence":"high","recommended_action":"Continue to maintain strong performance"}`                                                                                      | Pass                   | Pass                          |
| `{"CGPA": 7.6, "Internships": 1.0, "Projects": 1.0, "Workshops/Certifications": 1.0, "AptitudeTestScore": 71.0, "SoftSkillsRating": 4.2, "ExtracurricularActivities": "no", "PlacementTraining": "no", "SSC_Marks": 64.0, "HSC_Marks": 71.0}`           | `json {"risk_tier":"medium","flag_for_review":true,"primary_signal":"PlacementTraining is missing","confidence":"medium","recommended_action":"Enroll in placement training"} `                                                                               | Fail                   | Block                         |
| `{"CGPA": 7.4, "Internships": 2.0, "Projects": 3.0, "Workshops/Certifications": 3.0, "AptitudeTestScore": 86.0, "SoftSkillsRating": 4.2, "ExtracurricularActivities": "yes", "PlacementTraining": "yes", "SSC_Marks": 60.0, "HSC_Marks": 75.0}`         | `json {"risk_tier":"low","flag_for_review":false,"primary_signal":"Consistently strong readiness indicators across multiple metrics","confidence":"high","recommended_action":"Continue to leverage existing strengths and seek challenging opportunities"} ` | Fail                   | Block                         |
