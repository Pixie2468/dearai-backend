"""Graph schema definitions for the Dear AI knowledge graph."""

from enum import StrEnum

from graphrag_sdk import EntityType, GraphSchema, RelationType


class EntityLabel(StrEnum):
    """Entity labels used in the graph."""

    USER = "User"
    MOOD = "Mood"
    PERSON = "Person"
    TOPIC = "Topic"
    SESSION = "Session"


class RelationLabel(StrEnum):
    """Relation labels used in the graph."""

    FEELS = "FEELS"
    KNOWS = "KNOWS"
    FEELS_ABOUT = "FEELS_ABOUT"
    DISCUSSED = "DISCUSSED"
    PARTICIPATED_IN = "PARTICIPATED_IN"
    REVEALED_MOOD = "REVEALED_MOOD"
    DISCUSSED_TOPIC = "DISCUSSED_TOPIC"
    CAUSES_MOOD = "CAUSES_MOOD"


def create_graph_schema() -> GraphSchema:
    """Build and return the GraphSchema for the Dear AI knowledge graph."""
    return GraphSchema(
        entities=[
            EntityType(label=EntityLabel.USER, description="The human chatting with the bot"),
            EntityType(
                label=EntityLabel.MOOD,
                description="The emotional state of the user (e.g., Happy, Anxious, Frustrated)",
            ),
            EntityType(
                label=EntityLabel.PERSON,
                description="A friend, family member, or colleague in the user's life, or the user himself",
            ),
            EntityType(
                label=EntityLabel.TOPIC,
                description="A subject, a hobby, or event being discussed",
            ),
            EntityType(
                label=EntityLabel.SESSION,
                description="A specific chat session occurring on a specific date. MUST include the date in its ID",
            ),
        ],
        relations=[
            RelationType(
                label=RelationLabel.FEELS,
                description="The current mood the user is experiencing",
                patterns=[(EntityLabel.USER, EntityLabel.MOOD)],
            ),
            RelationType(
                label=RelationLabel.KNOWS,
                description="A person the user interacts with",
                patterns=[(EntityLabel.USER, EntityLabel.PERSON)],
            ),
            RelationType(
                label=RelationLabel.FEELS_ABOUT,
                description="How the user feels regarding a specific subject",
                patterns=[(EntityLabel.USER, EntityLabel.TOPIC)],
            ),
            RelationType(
                label=RelationLabel.DISCUSSED,
                description="A topic brought up in the conversation",
                patterns=[(EntityLabel.USER, EntityLabel.TOPIC)],
            ),
            RelationType(
                label=RelationLabel.PARTICIPATED_IN,
                description="Connects the user to a specific chat session",
                patterns=[(EntityLabel.USER, EntityLabel.SESSION)],
            ),
            RelationType(
                label=RelationLabel.REVEALED_MOOD,
                description="The mood the user expressed during the specific session",
                patterns=[(EntityLabel.SESSION, EntityLabel.MOOD)],
            ),
            RelationType(
                label=RelationLabel.DISCUSSED_TOPIC,
                description="What was talked about during this specific session",
                patterns=[(EntityLabel.SESSION, EntityLabel.TOPIC)],
            ),
            RelationType(
                label=RelationLabel.CAUSES_MOOD,
                description="When a topic triggers or causes a specific emotion during the session",
                patterns=[(EntityLabel.TOPIC, EntityLabel.MOOD)],
            ),
        ],
    )
