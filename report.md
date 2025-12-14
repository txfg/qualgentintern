Dear QualGent Team,
  After careful consideration of the popular frameworks provided, through Simular’s Agent S and Google’s Agent Development Kit,
I have decided to use the approach of a custom Supervisor-Planner-Executor Architecture designed from scratch.
I took this as it would provide me the most experience with understanding the structure for the problem, as well as give more flexibility.
My program is a lightweight, custom Python framework matching the architecture for this challenge:

  1.  **Supervisor** A Python-based orchestration loop that manages test state, enforces step limits, and handles error recovery.
  2.  **Planner**  Analyzes the screenshot and understands hierarchy for next moves.
  3.  **Executor (Tooling):** Pure Python ADB enabling precise device execution:  text input, taps, swipes

Rationale:
  Agent S and the google development kit are both very powerful, however, there is more transparancy with a custom model
as well as minimal overhead when running introducing latency, as the program is very intuitive.
Further this task introduced complexities for a standard model to understand when to value vision, or UI prompt
to make its next move, and many of these technical difficulties can be resolved much faster with a custom model.
Finally, I believe using an existing framework would give less of a personal gain in completing this project,
where I am now more capable of developing agentic frameworks, and naturally implementing existing ones.

Kind regards,
Fernando C Guzman
