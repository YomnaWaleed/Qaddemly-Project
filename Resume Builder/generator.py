import os
from dotenv import load_dotenv
from groq import Groq

class ResumeSectionGenerator:
    """
    Base class for generating and enhancing resume sections using Groq API.

    This class provides the foundation for resume section generation, handling API setup
    and input validation. Child classes define section-specific prompts, section type,
    and token limits.

    Attributes
    ----------
    api_key : str
        The Groq API key loaded from the .env file.
    client : Groq
        The Groq client initialized with the API key.
    """

    def __init__(self):
        """Initialize the ResumeSectionGenerator with API key and client."""
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in .env file!")
        self.client = Groq(api_key=self.api_key)

    def _validate_inputs(self, user_json, mode, existing_section):
        """
        Validate input parameters for section generation or enhancement.

        If mode is 'enhance' and existing_section is empty, switches to 'generate' mode.

        Parameters
        ----------
        user_json : dict
            User data with a 'user' key containing profile information.
        mode : str
            Operation mode ('generate' or 'enhance').
        existing_section : str or None
            Existing section text to enhance (optional for 'enhance' mode).

        Returns
        -------
        str
            Validated mode ('generate' or 'enhance').

        Raises
        ------
        ValueError
            If inputs are invalid or missing.
        """
        if not user_json.get("user"):
            raise ValueError("Input JSON must contain a 'user' key")
        if mode not in ["generate", "enhance"]:
            raise ValueError(f"Invalid mode '{mode}'")
        if mode == "enhance" and not existing_section:
            mode = "generate"
        return mode

    def generate_section(self, user_json, mode, existing_section, job_description):
        """
        Generate or enhance a resume section using the Groq API.

        Parameters
        ----------
        user_json : dict
            User data with a 'user' key containing profile information.
        mode : str
            Operation mode ('generate' or 'enhance').
        existing_section : str or None
            Existing section text to enhance (optional for 'enhance' mode).
        job_description : str or None
            Optional job description to tailor the section for ATS compatibility.

        Returns
        -------
        str
            Generated or enhanced section (paragraph for 'about_me', list for 'skills').

        Raises
        ------
        ValueError
            If API call fails or inputs are invalid.
        """
        mode = self._validate_inputs(user_json, mode, existing_section)
        user = user_json["user"]
        # Format user data into a string for the prompt
        skills = ", ".join([skill["name"] for skill in user.get("skills", [])]) or "None"
        educations = ", ".join([f"{edu['field_of_study']} at {edu['university']}" for edu in user.get("educations", [])]) or "None"
        certificates = ", ".join([cert["name"] for cert in user.get("certificates", [])]) or "None"
        languages = ", ".join([lang["name"] for lang in user.get("languages", [])]) or "None"
        experiences = ", ".join([f"{exp['job_title']} at {exp['company_name']}" for exp in user.get("experiences", [])]) or "None"
        user_str = f"Skills: {skills}; Education: {educations}; Certificates: {certificates}; Languages: {languages}; Experiences: {experiences}"
        job_description_str = job_description if job_description else "None"

        system_prompt, user_prompt_template = self._get_prompts(mode)
        user_prompt = user_prompt_template.format(
            user=user_str,
            job_description=job_description_str,
            existing_section=existing_section or "None"
        )

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise ValueError(f"Groq API error: {str(e)}")

    def _get_prompts(self, mode):
        """
        Abstract method to retrieve section-specific prompts.

        Must be implemented by child classes.

        Parameters
        ----------
        mode : str
            Operation mode ('generate' or 'enhance').

        Returns
        -------
        tuple
            System prompt and user prompt template.

        Raises
        ------
        NotImplementedError
            If not implemented by child class.
        """
        raise NotImplementedError("Child classes must implement _get_prompts")

class AboutMeGenerator(ResumeSectionGenerator):
    """
    Generator for 'About Me' resume sections.

    Handles generation and enhancement of the 'About Me' section with specific prompts
    and token limits.

    Attributes
    ----------
    section_type : str
        The section type handled by this class ('about_me').
    max_tokens : int
        Maximum token limit for the section (200).
    """

    def __init__(self):
        """Initialize the AboutMeGenerator with section type and token limit."""
        super().__init__()
        self.section_type = "about_me"
        self.max_tokens = 200

    def _get_prompts(self, mode):
        """
        Retrieve prompts for 'About Me' section.

        Parameters
        ----------
        mode : str
            Operation mode ('generate' or 'enhance').

        Returns
        -------
        tuple
            System prompt and user prompt template.

        Raises
        ------
        ValueError
            If mode is invalid.
        """
        prompts = {
            "generate": {
                "system": "You are an expert resume writer crafting natural, ATS-optimized summaries.",
                "user": """
                    Write a professional 'About Me' section as a single paragraph:
                    - Limit to 7 lines (~100 words).
                    - Highlight skills, certifications, education, and experiences from {user}.
                    - Use confident, natural tone with action verbs (e.g., 'specialize', 'excel').
                    - Tailor to {job_description} for ATS compatibility if provided.
                    - Return only the paragraph.
                """
            },
            "enhance": {
                "system": "You are an expert resume editor refining ATS-optimized summaries.",
                "user": """
                    Enhance this 'About Me' section as a single paragraph:
                    - Limit to 7 lines (~100 words).
                    - Retain skills, certifications, and experiences from {user}, improve clarity.
                    - Use confident tone with action verbs.
                    - Tailor to {job_description} for ATS compatibility if provided.
                    - Enhance {existing_section}.
                    - Return only the paragraph.
                """
            }
        }
        if mode not in prompts:
            raise ValueError(f"Invalid mode '{mode}'")
        return prompts[mode]["system"], prompts[mode]["user"]

class SkillsGenerator(ResumeSectionGenerator):
    """
    Generator for 'Skills' resume sections.

    Handles generation and enhancement of the 'Skills' section with specific prompts
    and token limits.

    Attributes
    ----------
    section_type : str
        The section type handled by this class ('skills').
    max_tokens : int
        Maximum token limit for the section (100).
    """

    def __init__(self):
        """Initialize the SkillsGenerator with section type and token limit."""
        super().__init__()
        self.section_type = "skills"
        self.max_tokens = 100

    def _get_prompts(self, mode):
        """
        Retrieve prompts for 'Skills' section.

        Parameters
        ----------
        mode : str
            Operation mode ('generate' or 'enhance').

        Returns
        -------
        tuple
            System prompt and user prompt template.

        Raises
        ------
        ValueError
            If mode is invalid.
        """
        prompts = {
            "generate": {
                "system": "You are an expert resume writer creating ATS-optimized skill lists.",
                "user": """
                    - Generate a single line of 8–12 skills with 60% hard skills and 40% soft skills.
                    - Base skills on {user} if provided, otherwise assume software engineering skills.
                    - Tailor skills to {job_description} if provided, otherwise use general technical and soft skills.
                    - Use clear, ATS-friendly terms.
                    - Separate skills with commas and spaces.
                    - Exclude all additional text, explanations, headings, bullet points, or formatting.
                    - Avoid extra spaces, newlines, or punctuation like ';', '.', or colons.
                    - Return only the comma-separated list of skills.
                    - Do not return any other text except the skills.
                    - Assume missing requirements, don't ask or mention that is something required to do your job, mention ONLY SKILLS!
                    - Do not even mention "Based in your information" or any intro, mention only skills directly.
                """
            },
            "enhance": {
                "system": "You are an expert resume editor refining ATS-optimized skill lists.",
                "user": """
                    - Enhance {existing_section} into a single line of 8–12 skills with 60% hard skills and 40% soft skills.
                    - Base skills on {user} if provided, otherwise assume software engineering skills.
                    - Tailor skills to {job_description} if provided, otherwise use general technical and soft skills.
                    - Use clear, ATS-friendly terms.
                    - Separate skills with commas and spaces.
                    - Exclude all additional text, explanations, headings, bullet points, or formatting.
                    - Avoid extra spaces, newlines, or punctuation like ';', '.', or colons.
                    - Return only the comma-separated list of skills.
                    - Do not return any other text except the skills.
                    - Assume missing requirements, don't ask or mention that is something required to do your job, mention ONLY SKILLS!
                    - Do not even mention "Based in your information" or any intro, mention only skills directly.
                """
            }
        }
        if mode not in prompts:
            raise ValueError(f"Invalid mode '{mode}'")
        return prompts[mode]["system"], prompts[mode]["user"]

# Example usage
if __name__ == "__main__":
    import json

    # Sample user data
    test_file_path = "data/test.json"
    with open(test_file_path, "r", encoding="utf-8") as file:
        user_data = json.load(file)

    # Sample job description
    job_description = "We are seeking a skilled Backend Developer proficient in Node.js, TypeScript, and PostgreSQL to build scalable APIs and integrate AI-driven solutions. The role requires strong problem-solving skills and experience with Spring Boot or Java-based frameworks."

    # Sample existing sections for enhancement
    existing_about_me = ""
    existing_skills = ""

    # Initialize generators
    about_me_gen = AboutMeGenerator()
    skills_gen = SkillsGenerator()

    # Generate About Me
    about_me = about_me_gen.generate_section(user_data, "generate", None, job_description)
    print("Generated About Me:", about_me)

    # Enhance About Me (should switch to generate due to empty existing_section)
    enhanced_about_me = about_me_gen.generate_section(user_data, "enhance", existing_about_me, job_description)
    print("Enhanced About Me:", enhanced_about_me)

    # Generate Skills
    skills = skills_gen.generate_section(user_data, "generate", None, job_description)
    print("Generated Skills:", skills)

    # Enhance Skills (should switch to generate due to empty existing_section)
    enhanced_skills = skills_gen.generate_section(user_data, "enhance", existing_skills, job_description)
    skills = skills_gen.generate_section(user_data, "generate", None, job_description).split(', ')
    if skills[-1][-1] == '.':
        skills[-1] = skills[-1][:-1]
    print("Generated Skills:", skills)

    # Enhance Skills (should switch to generate due to empty existing_section)
    enhanced_skills = skills_gen.generate_section(user_data, "enhance", existing_skills, job_description).split(', ')
    if enhanced_skills[-1][-1] == '.':
        enhanced_skills[-1] = enhanced_skills[-1][:-1]
    print("Enhanced Skills:", enhanced_skills)