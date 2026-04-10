"""Shared configuration values and prompt templates for the experiment package."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PACKAGE_DIR / "outputs"


OLLAMA_DEFAULT_BASE_URL = "OLLAMA_BASE_URL"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"


MODEL_QWEN_2_5_72B = "qwen2.5:72b"
MODEL_GPT_OSS_120B = "gpt-oss:120b"
MODEL_OPENAI_GPT_5_NANO = "gpt-5-nano"
MODEL_OPENAI_GPT_5 = "gpt-5"
MODEL_OPENAI_GPT_5_1 = "gpt-5.1"


RAW_TRAIN_DATA = DATA_DIR / "TRAIN_RELEASE_3SEP2025" / "TRAIN_RELEASE_3SEP2025" / "train_subtask1.csv"
RAW_TEST_DATA = DATA_DIR / "TEST_RELEASE_5JAN2026" / "test_subtask1.csv"
TRAIN_DATA_EMOTION = DATA_DIR / "train_data_emotion.csv"


VA_TO_EMOTIONS = {
    (-2, 2): "Jittery, nervous",
    (-1, 2): "Somewhat jittery",
    (0, 2): "Active",
    (1, 2): "Somewhat lively",
    (2, 2): "Lively, enthusiastic",
    (-2, 1): "Very sad",
    (-1, 1): "Somewhat sad",
    (0, 1): "Neutral",
    (1, 1): "Somewhat happy",
    (2, 1): "Very happy",
    (-2, 0): "Sluggish, tired",
    (-1, 0): "Somewhat sluggish",
    (0, 0): "Quiet",
    (1, 0): "Somewhat content",
    (2, 0): "Content, calm",
}

EMOTIONS_TO_VA = {value: key for key, value in VA_TO_EMOTIONS.items()}
ALLOWED_EMOTIONS = tuple(VA_TO_EMOTIONS.values())


PROMPT_15SHOT = """
You are an expert in human emotions. Below is a list of short texts, where each text describes how a person feels today. Your task is to assign exactly one emotion from the allowed list to each text - the emotion that best matches the overall feeling expressed.
Instructions:
- If the text does not explicitly describe feelings, choose the emotion that best fits the emotional state implied by the text.
- Preserve the original order of the texts.
- Each text must appear exactly once in the output - no duplicates and none omitted.
- Present the result in a plain Python-friendly dictionary format, without any explanations or comments.
Emotions:
{{"Jittery, nervous", "Somewhat jittery", "Active", "Somewhat lively", "Lively, enthusiastic", "Very sad", "Somewhat sad", "Neutral", "Somewhat happy", "Very happy", "Sluggish, tired", "Somewhat sluggish", "Quiet", "Somewhat content", "Content, calm"}}
Examples:
1. I have been feeling somewhat down . I am trying to be more productive and get things done . I am having trouble finding the energy and motivation . -> "Somewhat sluggish"
2. Calm , Content , Happy , Relaxed -> "Very happy"
3. I feel okay today , but of course I’m off today . Yesterday was a long , bad day . I got a good nights rest . I feel so much better today . -> "Neutral"
4. trapped , stuck , anxious , confused , lost . -> "Very sad"
5. I am all nervous . I just took a test . I am worried and scared of the results . -> "Jittery, nervous"
6. A little nauseous but energetic . I cleaned for hours and got a good workout from that . -> "Active"
7. calm , content , relaxed , tired , still -> "Somewhat content"
8. I am feeling drained and irritable . I am exhausted and bored because I am too tired to do anything . -> "Somewhat sad"
9. energetic , happy , smiling , excited , ready -> "Somewhat lively"
10. Chill , Relaxed , Calm , Mellow , Grateful -> "Content, calm"
11. I am tired and having anxiety . I am a little jittery and unable to relax . -> "Somewhat jittery"
12. Lively , Active , Relaxed , Charming , Energetic -> "Lively, enthusiastic"
13. I am happy to be here and looking forward to the week . -> "Somewhat happy"
14. Tired , Dehydrated , Sluggish , Sick , Numb -> "Sluggish, tired"
15. Today it’s raining and I just want to lay down all day , relax and watch movies . -> "Quiet"
Output format example:
{{text_id1: "emotion1", text_id2: "emotion2", text_id3: "emotion3"}}
List of texts:
{}
"""


PROMPT_15SHOT_ESSAYS = """
You are an expert in human emotions. Below is a list of texts, where each text describes how a person feels today. These texts are typically longer diary-style entries.
Your task is to assign exactly one emotion from the allowed list to each text.
Instructions:
- Focus on the overall emotional state of the full text.
- Preserve the original order of the texts.
- Each text must appear exactly once in the output.
- Present the result as a plain Python-friendly dictionary with no extra commentary.
Allowed emotions:
{{"Jittery, nervous", "Somewhat jittery", "Active", "Somewhat lively", "Lively, enthusiastic", "Very sad", "Somewhat sad", "Neutral", "Somewhat happy", "Very happy", "Sluggish, tired", "Somewhat sluggish", "Quiet", "Somewhat content", "Content, calm"}}
Examples:
1. I am all nervous ... election results ... -> "Jittery, nervous"
2. I am feeling better than yesterday ... low on my self esteem ... -> "Very sad"
3. I feel super sluggish because I might have eaten something bad ... -> "Sluggish, tired"
4. I have been feeling somewhat down ... trouble finding the energy ... -> "Somewhat sluggish"
5. I am feeling drained and irritable ... -> "Somewhat sad"
6. I am tired and having anxiety ... -> "Somewhat jittery"
7. A little nauseous but energetic ... -> "Active"
8. I feel okay today ... got a good nights rest ... -> "Neutral"
9. I am happy to be here and looking forward to the week ... -> "Somewhat happy"
10. Just hanging out at home, peaceful and content ... -> "Content, calm"
Output format example:
{{text_id1: "emotion1", text_id2: "emotion2"}}
List of texts:
{}
"""


PROMPT_15SHOT_WORDS = """
You are an expert in human emotions. Below is a list of texts, where each text is a short feeling-word list.
Your task is to assign exactly one emotion from the allowed list to each text.
Instructions:
- Infer the best matching overall emotion from the listed words.
- Preserve the original order of the texts.
- Each text must appear exactly once in the output.
- Present the result as a plain Python-friendly dictionary with no extra commentary.
Allowed emotions:
{{"Jittery, nervous", "Somewhat jittery", "Active", "Somewhat lively", "Lively, enthusiastic", "Very sad", "Somewhat sad", "Neutral", "Somewhat happy", "Very happy", "Sluggish, tired", "Somewhat sluggish", "Quiet", "Somewhat content", "Content, calm"}}
Examples:
1. Calm , Indifferent , Present , Mindful , Chill -> "Neutral"
2. Happy , Comforted , Pampered , Loved , Joyful -> "Somewhat happy"
3. Tired , Sore , Sleepy , Heavy , Groggy -> "Sluggish, tired"
4. Aware , Antsy , Impatient , Gassy , Full -> "Somewhat jittery"
5. Chill , Calm , Satisfied , Warm , Leisurely -> "Content, calm"
6. Energized , Active , Mobile , Motivated , Happy -> "Active"
Output format example:
{{text_id1: "emotion1", text_id2: "emotion2"}}
List of texts:
{}
"""


PROMPT_USER_AWARE_STATIC_EMOTION = """
You are an expert in human emotions. Below is a chronological sequence of short texts written by the same user, each describing how they felt on a particular day.
Your task is to assign exactly one emotion from the allowed list to each text - the emotion that best matches the overall feeling expressed.
The user has a personal, consistent way of expressing emotions. Learn from the previously labeled examples how this specific user tends to describe their emotional states, and apply this understanding when labeling the new texts.
Instructions:
- If the text does not explicitly describe feelings, choose the emotion that best fits the emotional state implied by the text.
- All texts in the evaluation set come from the same user as the examples.
- Preserve the original order of the texts.
- Each text must appear exactly once in the output.
- Present the result in a plain Python-friendly dictionary format, without any explanations or comments.
Allowed emotions:
{{"Jittery, nervous", "Somewhat jittery", "Active", "Somewhat lively", "Lively, enthusiastic", "Very sad", "Somewhat sad", "Neutral", "Somewhat happy", "Very happy", "Sluggish, tired", "Somewhat sluggish", "Quiet", "Somewhat content", "Content, calm"}}
Previous texts with assigned emotions:
{train}
Sequence of texts for evaluation:
{predict}
"""


PROMPT_DYNAMIC = """
You are an expert in human emotions. Below is a chronological sequence of short texts written by the same user, each describing how they felt on a particular day.
Your task is to assign exactly one emotion from the allowed list to each text - the emotion that best matches the overall feeling expressed.
The user has a personal, consistent way of expressing emotions. Learn from the previously labeled examples how this specific user tends to describe their emotional states, and apply this understanding when labeling the new texts.
Important clarification:
The texts to be labeled are a direct chronological continuation of the previously labeled history.
Instructions:
- If the text does not explicitly describe feelings, choose the emotion that best fits the emotional state implied by the text.
- Preserve the original order of the texts.
- Each text must appear exactly once in the output.
- Present the result in a plain Python-friendly dictionary format, without any explanations or comments.
Allowed emotions:
{{"Jittery, nervous", "Somewhat jittery", "Active", "Somewhat lively", "Lively, enthusiastic", "Very sad", "Somewhat sad", "Neutral", "Somewhat happy", "Very happy", "Sluggish, tired", "Somewhat sluggish", "Quiet", "Somewhat content", "Content, calm"}}
Previously labeled history (earliest to latest):
{train}
labeled continuation (chronologically follows the history above):
{predict}
"""


FAKE_WORDS_HISTORY = [
    "Calm , Indifferent , Present , Mindful , Chill -> Neutral",
    "Happy , Comforted , Pampered , Loved , Joyful -> Somewhat happy",
    "Tired , Sore , Sleepy , Heavy , Groggy -> Sluggish, tired",
    "Aware , Antsy , Impatient , Gassy , Full -> Somewhat jittery",
    "Peaceful , Calm , Warm , Fluid , Curious -> Somewhat sluggish",
    "Warm , Tired , Unmotivated , Puddle , Full -> Somewhat sluggish",
    "Calm , Refreshed , Introspective , Enriched , Meager -> Somewhat happy",
    "Chill , Calm , Satisfied , Warm , Leisurely -> Content, calm",
    "Energized , Active , Mobile , Motivated , Happy -> Active",
    "Motivated , Proud , Settled , Talented , Fitted -> Neutral",
]


FAKE_ESSAYS_HISTORY = [
    "I have just been hanging out at home for a few days so I just feel calm and content . -> Content, calm",
    "I am feeling pretty calm and content after an easy shift . -> Somewhat content",
    "It's been a pretty good day so I am just happy and relaxed . -> Somewhat happy",
    "I stayed out late last night and now I am just tired . -> Somewhat sluggish",
    "I'm feeling okay now . It's a pretty day so that makes me feel happy . -> Neutral",
    "I feel calm and just sleepy . Today was quiet and uneventful . -> Quiet",
    "I have an exciting day today so I am pretty happy . -> Content, calm",
    "I got to leave work early and hang out with my friends so I am super happy . -> Very happy",
    "I had a fun time last night but did not sleep well so I feel super sluggish today . -> Somewhat sluggish",
    "I have been having a rough day and I just feel tired and very sluggish . -> Sluggish, tired",
]


PROMPT_USER_AWARE_STATIC_VALENCE = """
You are an expert in human emotions. Below is a chronological sequence of short texts written by the same user, each describing how they felt on a particular day.
Your task is to assign a single valence value to each text, using the scale from -2 to +2.
Valence scale:
-2 = clearly negative
-1 = moderately negative
 0 = neutral or mixed
+1 = moderately positive
+2 = clearly positive
The user has a personal, consistent way of expressing emotions. Learn from the previously labeled examples how this specific user tends to describe positivity and negativity.
Instructions:
- Prefer the weakest value that sufficiently fits the text unless there is clear evidence for stronger intensity.
- If the text does not explicitly describe feelings, infer the best fitting valence from the implied state.
- Preserve the original order of the texts.
- Each text must appear exactly once in the output.
- Present the result in a plain Python-friendly dictionary format, without any explanations or comments.
Previous texts with assigned valence:
{train}
Sequence of texts for evaluation:
{predict}
"""


PROMPT_USER_AWARE_STATIC_AROUSAL = """
You are an expert in human emotions. Below is a chronological sequence of short texts written by the same user, each describing how they felt on a particular day.
Your task is to assign a single arousal value to each text, using the scale from 0 to 2.
Arousal scale:
0 = low activation
1 = moderate activation
2 = high activation
The user has a personal, consistent way of expressing activation and intensity. Learn from the previously labeled examples how this specific user expresses calmness, moderate activation, and high activation.
Instructions:
- Prefer the weakest value that sufficiently fits the text unless there is clear evidence for stronger activation.
- Do not infer higher arousal solely from longer text.
- Preserve the original order of the texts.
- Each text must appear exactly once in the output.
- Present the result in a plain Python-friendly dictionary format, without any explanations or comments.
Previous texts with assigned arousal:
{train}
Sequence of texts for evaluation:
{predict}
"""


PROMPT_USER_AWARE_STATIC_VAL_AND_ARO = """
You are an expert in human emotions. Below is a chronological sequence of short texts written by the same user, each describing how they felt on a particular day.
Your task is to assign two numerical values to each text:
- valence from -2 to +2
- arousal from 0 to 2
Valence and arousal must be inferred independently.
Valence scale:
-2 = clearly negative
-1 = moderately negative
 0 = neutral or mixed
+1 = moderately positive
+2 = clearly positive
Arousal scale:
0 = low activation
1 = moderate activation
2 = high activation
The user has a personal, consistent way of expressing emotions. Learn from the previously labeled examples and apply this user-specific interpretation.
Instructions:
- Prefer the weakest value that sufficiently fits the text unless there is clear evidence for stronger intensity or activation.
- Preserve the original order of the texts.
- Each text must appear exactly once in the output.
- Present the result in a plain Python-friendly dictionary format, without any explanations or comments.
Output format example:
{{text_id1: {{valence: 1, arousal: 0}}, text_id2: {{valence: -1, arousal: 1}}}}
Previous texts with assigned valence and arousal:
{train}
Sequence of texts for evaluation:
{predict}
"""


USER_AWARE_PROMPTS = {
    "emotion": PROMPT_USER_AWARE_STATIC_EMOTION,
    "valence": PROMPT_USER_AWARE_STATIC_VALENCE,
    "arousal": PROMPT_USER_AWARE_STATIC_AROUSAL,
    "val_aro": PROMPT_USER_AWARE_STATIC_VAL_AND_ARO,
}
