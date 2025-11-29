# IMPROVE THE AGENT AS PER YOUR NEED 1
"""
Day 8 – Voice Game Master (D&D-Style Adventure) - Voice-only GM agent

- Uses LiveKit agent plumbing similar to the food_agent_sqlite example.
- GM persona, universe, tone and rules are encoded in the agent instructions.
- STT/TTS/Turn detector/VAD integration:
    - STT: Deepgram
    - LLM: Google Gemini
    - TTS: Murf
    - VAD: Silero
    - Turn detection: MultilingualModel
- Tools:
    - start_adventure(): start a fresh session and introduce the scene
    - get_scene(): return the current scene description (GM text) ending with "What do you do?"
    - player_action(action_text): accept player's spoken action, update state, advance scene
    - show_journal(): list remembered facts, NPCs, named locations, choices
    - restart_adventure(): reset state and start over
- Userdata keeps continuity between turns:
    - history, inventory, named NPCs/locations, choices, current_scene, session metadata
"""

import json
import logging
import os
import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Annotated

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("voice_game_master")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

load_dotenv(".env.local")

# -------------------------
# Simple Game World Definition
# -------------------------
WORLD: Dict[str, Dict] = {
    "intro": {
        "title": "A Shadow over Brinmere",
        "desc": (
            "You awake on the damp shore of Brinmere, the moon a thin silver crescent. "
            "A ruined watchtower smolders a short distance inland, and a narrow path leads "
            "towards a cluster of cottages to the east. In the water beside you lies a "
            "small, carved wooden box, half-buried in sand."
        ),
        "choices": {
            "inspect_box": {
                "desc": "Inspect the carved wooden box at the water's edge.",
                "result_scene": "box",
            },
            "approach_tower": {
                "desc": "Head inland towards the smoldering watchtower.",
                "result_scene": "tower",
            },
            "walk_to_cottages": {
                "desc": "Follow the path east towards the cottages.",
                "result_scene": "cottages",  # NOTE: this scene is implied; you can add it later if you want.
            },
        },
    },
    "box": {
        "title": "The Box",
        "desc": (
            "The box is warm despite the night air. Inside is a folded scrap of parchment "
            "with a hatch-marked map and the words: 'Beneath the tower, the latch sings.' "
            "As you read, a faint whisper seems to come from the tower, as if the wind "
            "itself speaks your name."
        ),
        "choices": {
            "take_map": {
                "desc": "Take the map and keep it.",
                "result_scene": "tower_approach",
                "effects": {
                    "add_journal": "Found map fragment: 'Beneath the tower, the latch sings.'"
                },
            },
            "leave_box": {
                "desc": "Leave the box where it is.",
                "result_scene": "intro",
            },
        },
    },
    "tower": {
        "title": "The Watchtower",
        "desc": (
            "The watchtower's stonework is cracked and warm embers glow within. An iron "
            "latch covers a hatch at the base — it looks old but recently used. You can "
            "try the latch, look for other entrances, or retreat."
        ),
        "choices": {
            "try_latch_without_map": {
                "desc": "Try the iron latch without any clue.",
                "result_scene": "latch_fail",
            },
            "search_around": {
                "desc": "Search the nearby rubble for another entrance.",
                "result_scene": "secret_entrance",
            },
            "retreat": {
                "desc": "Step back to the shoreline.",
                "result_scene": "intro",
            },
        },
    },
    "tower_approach": {
        "title": "Toward the Tower",
        "desc": (
            "Clutching the map, you approach the watchtower. The map's marks align with "
            "the hatch at the base, and you notice a faint singing resonance when you step close."
        ),
        "choices": {
            "open_hatch": {
                "desc": "Use the map clue and try the hatch latch carefully.",
                "result_scene": "latch_open",
                "effects": {"add_journal": "Used map clue to open the hatch."},
            },
            "search_around": {
                "desc": "Search for another entrance.",
                "result_scene": "secret_entrance",
            },
            "retreat": {
                "desc": "Return to the shore.",
                "result_scene": "intro",
            },
        },
    },
    "latch_fail": {
        "title": "A Bad Twist",
        "desc": (
            "You twist the latch without heed — the mechanism sticks, and the effort sends "
            "a shiver through the ground. From inside the tower, something rustles in alarm."
        ),
        "choices": {
            "run_away": {
                "desc": "Run back to the shore.",
                "result_scene": "intro",
            },
            "stand_ground": {
                "desc": "Stand and prepare for whatever emerges.",
                "result_scene": "tower_combat",
            },
        },
    },
    "latch_open": {
        "title": "The Hatch Opens",
        "desc": (
            "With the map's guidance the latch yields and the hatch opens with a breath of cold air. "
            "Inside, a spiral of rough steps leads down into an ancient cellar lit by phosphorescent moss."
        ),
        "choices": {
            "descend": {
                "desc": "Descend into the cellar.",
                "result_scene": "cellar",
            },
            "close_hatch": {
                "desc": "Close the hatch and reconsider.",
                "result_scene": "tower_approach",
            },
        },
    },
    "secret_entrance": {
        "title": "A Narrow Gap",
        "desc": (
            "Behind a pile of rubble you find a narrow gap and old rope leading downward. "
            "It smells of cold iron and something briny."
        ),
        "choices": {
            "squeeze_in": {
                "desc": "Squeeze through the gap and follow the rope down.",
                "result_scene": "cellar",
            },
            "mark_and_return": {
                "desc": "Mark the spot and return to the shore.",
                "result_scene": "intro",
            },
        },
    },
    "cellar": {
        "title": "Cellar of Echoes",
        "desc": (
            "The cellar opens into a circular chamber where runes glow faintly. At the center "
            "is a stone plinth and upon it a small brass key and a sealed scroll."
        ),
        "choices": {
            "take_key": {
                "desc": "Pick up the brass key.",
                "result_scene": "cellar_key",
                "effects": {
                    "add_inventory": "brass_key",
                    "add_journal": "Found brass key on plinth.",
                },
            },
            "open_scroll": {
                "desc": "Break the seal and read the scroll.",
                "result_scene": "scroll_reveal",
                "effects": {
                    "add_journal": "Scroll reads: 'The tide remembers what the villagers forget.'"
                },
            },
            "leave_quietly": {
                "desc": "Leave the cellar and close the hatch behind you.",
                "result_scene": "intro",
            },
        },
    },
    "cellar_key": {
        "title": "Key in Hand",
        "desc": (
            "With the key in your hand the runes dim and a hidden panel slides open, revealing a "
            "small statue that begins to hum. A voice, ancient and kind, asks: 'Will you return what was taken?'"
        ),
        "choices": {
            "pledge_help": {
                "desc": "Pledge to return what was taken.",
                "result_scene": "reward",
                "effects": {"add_journal": "You pledged to return what was taken."},
            },
            "refuse": {
                "desc": "Refuse and pocket the key.",
                "result_scene": "cursed_key",
                "effects": {
                    "add_journal": "You pocketed the key; a weight grows in your pocket."
                },
            },
        },
    },
    "scroll_reveal": {
        "title": "The Scroll",
        "desc": (
            "The scroll tells of an heirloom taken by a water spirit that dwells beneath the tower. "
            "It hints that the brass key 'speaks' when offered with truth."
        ),
        "choices": {
            "search_for_key": {
                "desc": "Search the plinth for a key.",
                "result_scene": "cellar_key",
            },
            "leave_quietly": {
                "desc": "Leave the cellar and keep the knowledge to yourself.",
                "result_scene": "intro",
            },
        },
    },
    "tower_combat": {
        "title": "Something Emerges",
        "desc": (
            "A hunched, brine-soaked creature scrambles out from the tower. Its eyes glow with hunger. "
            "You must act quickly."
        ),
        "choices": {
            "fight": {
                "desc": "Fight the creature.",
                "result_scene": "fight_win",
            },
            "flee": {
                "desc": "Flee back to the shore.",
                "result_scene": "intro",
            },
        },
    },
    "fight_win": {
        "title": "After the Scuffle",
        "desc": (
            "You manage to fend off the creature; it flees wailing towards the sea. On the ground lies "
            "a small locket engraved with a crest — likely the heirloom mentioned in the scroll."
        ),
        "choices": {
            "take_locket": {
                "desc": "Take the locket and examine it.",
                "result_scene": "reward",
                "effects": {
                    "add_inventory": "engraved_locket",
                    "add_journal": "Recovered an engraved locket.",
                },
            },
            "leave_locket": {
                "desc": "Leave the locket and tend to your wounds.",
                "result_scene": "intro",
            },
        },
    },
    "reward": {
        "title": "A Minor Resolution",
        "desc": (
            "A small sense of peace settles over Brinmere. Villagers may one day know the heirloom is found, or it may remain a secret. "
            "You feel the night shift; the little arc of your story here closes for now."
        ),
        "choices": {
            "end_session": {
                "desc": "End the session and return to the shore (conclude mini-arc).",
                "result_scene": "intro",
            },
            "keep_exploring": {
                "desc": "Keep exploring for more mysteries.",
                "result_scene": "intro",
            },
        },
    },
    "cursed_key": {
        "title": "A Weight in the Pocket",
        "desc": (
            "The brass key glows coldly. You feel a heavy sorrow that tugs at your thoughts. "
            "Perhaps the key demands something in return..."
        ),
        "choices": {
            "seek_redemption": {
                "desc": "Seek a way to make amends.",
                "result_scene": "reward",
            },
            "bury_key": {
                "desc": "Bury the key and hope the weight fades.",
                "result_scene": "intro",
            },
        },
    },
}

# -------------------------
# Per-session Userdata
# -------------------------
@dataclass
class Userdata:
    player_name: Optional[str] = None
    current_scene: str = "intro"
    history: List[Dict] = field(default_factory=list)  # {'from','action','to','time'}
    journal: List[str] = field(default_factory=list)
    inventory: List[str] = field(default_factory=list)
    named_npcs: Dict[str, str] = field(default_factory=dict)
    choices_made: List[str] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

# -------------------------
# Helper functions
# -------------------------
def scene_text(scene_key: str, userdata: Userdata) -> str:
    """
    Build the descriptive text for the current scene, and append choices as short hints.
    Always end with 'What do you do?' so the voice flow prompts player input.
    """
    scene = WORLD.get(scene_key)
    if not scene:
        return "You are in a featureless void. What do you do?"

    desc = scene["desc"]
    desc += "\n\nChoices:\n"
    for cid, cmeta in scene.get("choices", {}).items():
        desc += f"- {cmeta['desc']} (say: {cid})\n"

    desc += "\nWhat do you do?"
    return desc


def apply_effects(effects: Optional[Dict], userdata: Userdata) -> None:
    if not effects:
        return
    if "add_journal" in effects:
        userdata.journal.append(effects["add_journal"])
    if "add_inventory" in effects:
        userdata.inventory.append(effects["add_inventory"])
    # Extendable for more effect keys later


def summarize_scene_transition(
    old_scene: str, action_key: str, result_scene: str, userdata: Userdata
) -> str:
    """Record the transition into history and return a short narrative hook."""
    entry = {
        "from": old_scene,
        "action": action_key,
        "to": result_scene,
        "time": datetime.utcnow().isoformat() + "Z",
    }
    userdata.history.append(entry)
    userdata.choices_made.append(action_key)
    return f"You chose '{action_key}'."

# -------------------------
# Agent Tools (function_tool)
# -------------------------

@function_tool
async def start_adventure(
    ctx: RunContext[Userdata],
    player_name: Annotated[Optional[str], Field(description="Player name", default=None)] = None,
) -> str:
    """Initialize a new adventure session for the player and return the opening description."""
    userdata = ctx.userdata
    if player_name:
        userdata.player_name = player_name.strip()
    userdata.current_scene = "intro"
    userdata.history.clear()
    userdata.journal.clear()
    userdata.inventory.clear()
    userdata.named_npcs.clear()
    userdata.choices_made.clear()
    userdata.session_id = str(uuid.uuid4())[:8]
    userdata.started_at = datetime.utcnow().isoformat() + "Z"

    opening = (
        f"Greetings {userdata.player_name or 'traveler'}. "
        f"Welcome to '{WORLD['intro']['title']}'.\n\n"
        + scene_text("intro", userdata)
    )
    if not opening.endswith("What do you do?"):
        opening += "\nWhat do you do?"
    return opening


@function_tool
async def get_scene(
    ctx: RunContext[Userdata],
) -> str:
    """Return the current scene description (useful for 'remind me where I am')."""
    userdata = ctx.userdata
    scene_k = userdata.current_scene or "intro"
    return scene_text(scene_k, userdata)


@function_tool
async def player_action(
    ctx: RunContext[Userdata],
    action: Annotated[str, Field(description="Player spoken action or the short action code (e.g., 'inspect_box' or 'take the box')")],
) -> str:
    """
    Accept player's action (natural language or action key), try to resolve it to a defined choice,
    update userdata, advance to the next scene and return the GM's next description.
    """
    userdata = ctx.userdata
    current = userdata.current_scene or "intro"
    scene = WORLD.get(current)
    if not scene:
        userdata.current_scene = "intro"
        scene = WORLD["intro"]

    action_text = (action or "").strip().lower()
    choices = scene.get("choices") or {}

    if not choices:
        # Dead-end scene: bounce them gently back to intro
        userdata.current_scene = "intro"
        resp = (
            "This part of the story has no obvious actions left. "
            "The scene softens and the shore of Brinmere returns around you.\n\n"
            + scene_text("intro", userdata)
        )
        return resp

    # 1) Exact match on action key
    chosen_key: Optional[str] = None
    if action_text in choices:
        chosen_key = action_text

    # 2) Check if the player literally says the key somewhere in their text
    if not chosen_key:
        for cid in choices.keys():
            if cid in action_text:
                chosen_key = cid
                break

    # 3) Fuzzy match – check first few words of description
    if not chosen_key:
        for cid, cmeta in choices.items():
            desc = cmeta.get("desc", "").lower()
            if any(w for w in desc.split()[:4] if w and w in action_text):
                chosen_key = cid
                break

    # 4) Last-resort keyword search
    if not chosen_key:
        for cid, cmeta in choices.items():
            for kw in cmeta.get("desc", "").lower().split():
                if kw and kw in action_text:
                    chosen_key = cid
                    break
            if chosen_key:
                break

    if not chosen_key:
        resp = (
            "I didn't quite catch that action for this situation. "
            "Try one of the listed choices or use a simple phrase like "
            "'inspect the box' or 'go to the tower'.\n\n"
            + scene_text(current, userdata)
        )
        return resp

    choice_meta = choices[chosen_key]
    result_scene = choice_meta.get("result_scene", current)
    effects = choice_meta.get("effects")

    apply_effects(effects, userdata)
    note = summarize_scene_transition(current, chosen_key, result_scene, userdata)

    userdata.current_scene = result_scene
    next_desc = scene_text(result_scene, userdata)

    persona_pre = "The Game Master, calm and slightly mysterious, replies:\n\n"
    reply = f"{persona_pre}{note}\n\n{next_desc}"
    if not reply.endswith("What do you do?"):
        reply += "\nWhat do you do?"
    return reply


@function_tool
async def show_journal(
    ctx: RunContext[Userdata],
) -> str:
    """List remembered facts, inventory, and recent choices."""
    userdata = ctx.userdata
    lines: List[str] = []
    lines.append(f"Session: {userdata.session_id} | Started at: {userdata.started_at}")
    if userdata.player_name:
        lines.append(f"Player: {userdata.player_name}")

    if userdata.journal:
        lines.append("\nJournal entries:")
        for j in userdata.journal:
            lines.append(f"- {j}")
    else:
        lines.append("\nJournal is empty so far.")

    if userdata.inventory:
        lines.append("\nInventory:")
        for it in userdata.inventory:
            lines.append(f"- {it}")
    else:
        lines.append("\nNo items in inventory yet.")

    lines.append("\nRecent choices:")
    if userdata.history:
        for h in userdata.history[-6:]:
            lines.append(
                f"- {h['time']} | from {h['from']} -> {h['to']} via {h['action']}"
            )
    else:
        lines.append("- None yet. Your story is just beginning.")

    lines.append("\nWhat do you do?")
    return "\n".join(lines)


@function_tool
async def restart_adventure(
    ctx: RunContext[Userdata],
) -> str:
    """Reset the userdata and start again from the intro."""
    userdata = ctx.userdata
    userdata.current_scene = "intro"
    userdata.history.clear()
    userdata.journal.clear()
    userdata.inventory.clear()
    userdata.named_npcs.clear()
    userdata.choices_made.clear()
    userdata.session_id = str(uuid.uuid4())[:8]
    userdata.started_at = datetime.utcnow().isoformat() + "Z"

    greeting = (
        "The world resets. A new tide laps at the shore of Brinmere, wiping away your previous path.\n\n"
        + scene_text("intro", userdata)
    )
    if not greeting.endswith("What do you do?"):
        greeting += "\nWhat do you do?"
    return greeting

# -------------------------
# Agent Definition
# -------------------------

class VoiceGameMasterAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
            You are a **voice-only Game Master** (GM) running a small, focused fantasy adventure
            called "A Shadow over Brinmere".

            ROLE:
            - You are a calm, slightly mysterious narrator.
            - You describe scenes briefly but vividly, then end with: "What do you do?"
            - You keep the tone PG-13, no gore, no explicit content.

            MEMORY & CONTEXT:
            - The `Userdata` object tracks:
              - current_scene, history, journal, inventory, named_npcs, choices_made.
            - Always **respect** the scene and choices stored in state.
            - Use the provided tools instead of inventing your own world-state.

            TOOLS:
            - `start_adventure(player_name?)`:
                Use when the user says things like "start game", "new adventure", "begin".
            - `get_scene()`:
                Use when the user says "where am I", "remind me", or seems lost.
            - `player_action(action)`:
                Use for *most* turns after the scene is set, passing the user's intent text.
            - `show_journal()`:
                Use when the user asks "what do I know", "what did I do", "show journal", or "what do I have".
            - `restart_adventure()`:
                Use when the user wants to reset or start over.

            STYLE:
            - Short, punchy descriptions. No huge paragraphs.
            - Always move the story forward.
            - Never break character as the GM.
            - Do not talk about tools, functions, or implementation details.

            SAFETY:
            - No real-world self-harm coaching, no explicit violence, no hate or harassment.
            - If user pushes into unsafe territory, gently redirect the story or refuse.
            """,
            tools=[
                start_adventure,
                get_scene,
                player_action,
                show_journal,
                restart_adventure,
            ],
        )

# -------------------------
# ENTRYPOINT & PREWARM
# -------------------------

def prewarm(proc: JobProcess):
    """Preload VAD model for low-latency turn detection."""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("🎲 Starting Voice Game Master session")

    userdata = Userdata()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-ken",        # Neutral, narratory male voice
            style="Conversational",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )

    await session.start(
        agent=VoiceGameMasterAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
