"""
FalkorDB graph schema definitions and seed data for Dear AI.

Defines the knowledge graph structure used by the Graph RAG pipeline:
- User context graph: people, topics, emotions, events from conversations
- Mental health knowledge base: coping strategies, therapeutic techniques, resources

Usage:
    from app.services.context.graph_schema import ensure_schema, seed_knowledge_base
    await ensure_schema(graph)
    await seed_knowledge_base(graph)
"""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Index / Constraint creation
# ---------------------------------------------------------------------------

_INDEXES = [
    # User context nodes
    "CREATE INDEX IF NOT EXISTS FOR (u:User) ON (u.user_id)",
    "CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.name)",
    "CREATE INDEX IF NOT EXISTS FOR (t:Topic) ON (t.name)",
    "CREATE INDEX IF NOT EXISTS FOR (e:Emotion) ON (e.name)",
    "CREATE INDEX IF NOT EXISTS FOR (ev:Event) ON (ev.description)",
    # Knowledge base nodes
    "CREATE INDEX IF NOT EXISTS FOR (cs:CopingStrategy) ON (cs.name)",
    "CREATE INDEX IF NOT EXISTS FOR (r:Resource) ON (r.name)",
    "CREATE INDEX IF NOT EXISTS FOR (tt:TherapyTechnique) ON (tt.name)",
    "CREATE INDEX IF NOT EXISTS FOR (c:Condition) ON (c.name)",
]


async def ensure_schema(graph) -> None:
    """Create indexes on the FalkorDB graph if they don't already exist."""
    for idx_query in _INDEXES:
        try:
            await graph.query(idx_query)
        except Exception as exc:
            # FalkorDB may not support IF NOT EXISTS on all versions;
            # swallow duplicate-index errors gracefully.
            logger.debug("Index creation note: %s", exc)
    logger.info("Graph schema indexes ensured.")


# ---------------------------------------------------------------------------
# Mental Health Knowledge Base seed data
# ---------------------------------------------------------------------------

_SEED_CYPHER = """
// --- Conditions ---
MERGE (anxiety:Condition {name: 'Anxiety'})
SET anxiety.description = 'Persistent worry, nervousness, or unease about something with an uncertain outcome'

MERGE (depression:Condition {name: 'Depression'})
SET depression.description = 'Persistent feelings of sadness, hopelessness, and loss of interest'

MERGE (stress:Condition {name: 'Stress'})
SET stress.description = 'Mental or emotional strain resulting from demanding circumstances'

MERGE (loneliness:Condition {name: 'Loneliness'})
SET loneliness.description = 'Distressing feeling of being alone or separated'

MERGE (grief:Condition {name: 'Grief'})
SET grief.description = 'Deep sorrow especially caused by loss'

MERGE (anger:Condition {name: 'Anger Management'})
SET anger.description = 'Difficulty controlling intense feelings of displeasure'

MERGE (sleep_issues:Condition {name: 'Sleep Issues'})
SET sleep_issues.description = 'Problems falling asleep, staying asleep, or poor sleep quality'

MERGE (self_esteem:Condition {name: 'Low Self-Esteem'})
SET self_esteem.description = 'Negative perception of self-worth and capabilities'

// --- Coping Strategies ---
MERGE (breathing:CopingStrategy {name: 'Deep Breathing'})
SET breathing.description = 'Slow, deep breaths to activate the parasympathetic nervous system',
    breathing.steps = '1. Inhale slowly through your nose for 4 counts. 2. Hold for 4 counts. 3. Exhale slowly through your mouth for 6 counts. 4. Repeat 5-10 times.'

MERGE (grounding:CopingStrategy {name: '5-4-3-2-1 Grounding'})
SET grounding.description = 'Sensory awareness technique to anchor yourself in the present',
    grounding.steps = 'Notice 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste.'

MERGE (journaling:CopingStrategy {name: 'Journaling'})
SET journaling.description = 'Writing down thoughts and feelings to process emotions',
    journaling.steps = 'Set a timer for 10-15 minutes. Write freely about what you are feeling without judging.'

MERGE (exercise:CopingStrategy {name: 'Physical Exercise'})
SET exercise.description = 'Regular physical activity to boost mood and reduce stress',
    exercise.steps = 'Start with a 15-minute walk. Gradually build to 30 minutes of moderate activity most days.'

MERGE (mindfulness:CopingStrategy {name: 'Mindfulness Meditation'})
SET mindfulness.description = 'Non-judgmental awareness of present-moment experiences',
    mindfulness.steps = 'Find a quiet place. Focus on your breath. When your mind wanders, gently bring it back.'

MERGE (progressive:CopingStrategy {name: 'Progressive Muscle Relaxation'})
SET progressive.description = 'Systematically tensing and relaxing muscle groups',
    progressive.steps = 'Tense each muscle group for 5 seconds, then relax for 30 seconds. Start from toes, work up.'

MERGE (social:CopingStrategy {name: 'Social Connection'})
SET social.description = 'Reaching out to trusted friends, family, or support groups',
    social.steps = 'Identify one person you trust. Send a message or make a call. Be honest about how you feel.'

MERGE (routine:CopingStrategy {name: 'Healthy Routine'})
SET routine.description = 'Establishing consistent daily habits for stability',
    routine.steps = 'Set regular sleep/wake times. Plan meals. Include at least one enjoyable activity daily.'

MERGE (gratitude:CopingStrategy {name: 'Gratitude Practice'})
SET gratitude.description = 'Intentionally focusing on things to be thankful for',
    gratitude.steps = 'Each evening, write down 3 things you are grateful for, no matter how small.'

MERGE (self_compassion:CopingStrategy {name: 'Self-Compassion'})
SET self_compassion.description = 'Treating yourself with the same kindness you would offer a friend',
    self_compassion.steps = 'When you notice self-criticism, pause and ask: What would I say to a friend in this situation?'

// --- Therapy Techniques ---
MERGE (cbt:TherapyTechnique {name: 'Cognitive Behavioral Therapy'})
SET cbt.description = 'Identifying and challenging negative thought patterns',
    cbt.abbreviation = 'CBT'

MERGE (dbt:TherapyTechnique {name: 'Dialectical Behavior Therapy'})
SET dbt.description = 'Skills for emotional regulation, distress tolerance, and interpersonal effectiveness',
    dbt.abbreviation = 'DBT'

MERGE (act:TherapyTechnique {name: 'Acceptance and Commitment Therapy'})
SET act.description = 'Accepting difficult emotions while committing to values-based action',
    act.abbreviation = 'ACT'

// --- Crisis Resources ---
MERGE (crisis_line:Resource {name: 'Crisis Helpline'})
SET crisis_line.description = 'Immediate support for mental health emergencies',
    crisis_line.contact = '988 Suicide & Crisis Lifeline: Call or text 988',
    crisis_line.type = 'crisis'

MERGE (vandrevala:Resource {name: 'Vandrevala Foundation'})
SET vandrevala.description = 'India mental health helpline available 24/7',
    vandrevala.contact = '1860-2662-345',
    vandrevala.type = 'crisis'

MERGE (icas:Resource {name: 'iCall'})
SET icas.description = 'Psychosocial helpline by TISS Mumbai',
    icas.contact = '9152987821',
    icas.type = 'counseling'

// --- Relationships: Condition -> CopingStrategy ---
MERGE (anxiety)-[:HELPED_BY]->(breathing)
MERGE (anxiety)-[:HELPED_BY]->(grounding)
MERGE (anxiety)-[:HELPED_BY]->(mindfulness)
MERGE (anxiety)-[:HELPED_BY]->(progressive)

MERGE (depression)-[:HELPED_BY]->(exercise)
MERGE (depression)-[:HELPED_BY]->(social)
MERGE (depression)-[:HELPED_BY]->(routine)
MERGE (depression)-[:HELPED_BY]->(gratitude)
MERGE (depression)-[:HELPED_BY]->(journaling)

MERGE (stress)-[:HELPED_BY]->(breathing)
MERGE (stress)-[:HELPED_BY]->(exercise)
MERGE (stress)-[:HELPED_BY]->(mindfulness)
MERGE (stress)-[:HELPED_BY]->(journaling)

MERGE (loneliness)-[:HELPED_BY]->(social)
MERGE (loneliness)-[:HELPED_BY]->(journaling)
MERGE (loneliness)-[:HELPED_BY]->(gratitude)

MERGE (grief)-[:HELPED_BY]->(journaling)
MERGE (grief)-[:HELPED_BY]->(social)
MERGE (grief)-[:HELPED_BY]->(self_compassion)

MERGE (anger)-[:HELPED_BY]->(breathing)
MERGE (anger)-[:HELPED_BY]->(progressive)
MERGE (anger)-[:HELPED_BY]->(mindfulness)

MERGE (sleep_issues)-[:HELPED_BY]->(routine)
MERGE (sleep_issues)-[:HELPED_BY]->(progressive)
MERGE (sleep_issues)-[:HELPED_BY]->(mindfulness)

MERGE (self_esteem)-[:HELPED_BY]->(self_compassion)
MERGE (self_esteem)-[:HELPED_BY]->(gratitude)
MERGE (self_esteem)-[:HELPED_BY]->(journaling)

// --- Relationships: Condition -> TherapyTechnique ---
MERGE (anxiety)-[:TREATED_BY]->(cbt)
MERGE (anxiety)-[:TREATED_BY]->(act)
MERGE (depression)-[:TREATED_BY]->(cbt)
MERGE (depression)-[:TREATED_BY]->(act)
MERGE (anger)-[:TREATED_BY]->(dbt)
MERGE (self_esteem)-[:TREATED_BY]->(cbt)
"""


async def seed_knowledge_base(graph) -> None:
    """Populate the graph with the mental health knowledge base.

    This is idempotent thanks to MERGE -- safe to call on every startup.
    """
    try:
        await graph.query(_SEED_CYPHER)
        logger.info("Mental health knowledge base seeded successfully.")
    except Exception:
        logger.exception("Failed to seed knowledge base.")
