import os
from dotenv import load_dotenv
from agent import create_agent
import argparse

# Load environment variables if using .env file
load_dotenv()

def display_welcome():
    """Display welcome message and sample queries"""
    print("\n" + "="*80)
    print("Welcome to the Mondelez Market Analysis System")
    print("="*80)
    print("\nThis system can analyze market share and penetration data for Mondelez cookie brands.")
    print("\nSample Queries You Can Ask:")
    print("1. Can you compare monthly Market share of Oreo with Penetration data for first 7 months of 2025?")
    print("2. How does Oreo's market share in the Northeast compare with ChipsAhoy?")
    print("3. Which age group has the highest penetration for belVita and how has it changed over time?")
    print("4. Forecast Ritz's market share for the next 3 months based on current trends.")
    print("5. Compare NutterButter's performance in the West region with its national average.")
    print("6. Which Mondelez brand showed the highest growth in market share in the first half of 2025?")
    print("7. Is there a correlation between Oreo's price and market share?")
    print("\nType 'exit' to quit or 'help' for more information.")
    print("="*80 + "\n")

def run_interactive_mode(agent):
    """Run the agent in interactive mode"""
    display_welcome()
    
    while True:
        # Get query from user
        query = input("\nEnter your query (or 'exit' to quit, 'help' for assistance): ")
        
        if query.lower() == 'exit':
            print("Thank you for using the CPG Market Analysis System. Goodbye!")
            break
        elif query.lower() == 'help':
            display_help()
            continue
                
        # Run the agent with user query
        print("\nProcessing your query... This may take a moment.")
        result = agent.run(query)
        
        print("\nAnalysis Results:")
        print("-" * 80)
        print(result)
        print("-" * 80)

def display_help():
    """Display help information"""
    print("\n" + "="*80)
    print("HELP: Mondelez Market Analysis System")
    print("="*80)
    print("\nYou can ask questions about:")
    print("- Market share data for Mondelez brands")
    print("- Penetration data across demographics")
    print("- Brand comparisons and competitor analysis")
    print("- Regional performance differences")
    print("- Forecasts and trend analysis")
    print("\nAvailable Mondelez brands: Oreo, ChipsAhoy, Ritz, belVita, NutterButter")
    print("Available regions: Northeast, Southeast, Midwest, West, Southwest")
    print("Available age groups: 18-24, 25-34, 35-44, 45-54, 55+")
    print("="*80 + "\n")

def process_single_query(agent, query):
    """Process a single query and return the result"""
    print(f"Processing query: {query}")
    result = agent.run(query)
    print("\nAnalysis Results:")
    print("-" * 80)
    print(result)
    print("-" * 80)
    return result

def main():
    """Main function to run the application"""
    parser = argparse.ArgumentParser(description='CPG Market Analysis System')
    parser.add_argument('--query', '-q', type=str, help='Single query to process')
    parser.add_argument('--model', '-m', type=str, default='llama3-8b-8192', 
                        help='LLM model to use (default: llama3-8b-8192)')
    parser.add_argument('--temperature', '-t', type=float, default=0.2,
                        help='Temperature setting for generation (default: 0.2)')
    args = parser.parse_args()
    
    try:
        # Create agent with specified parameters
        agent = create_agent(
            llm_model=args.model,
            temperature=args.temperature
        )
        
        # Either process a single query or run in interactive mode
        if args.query:
            process_single_query(agent, args.query)
        else:
            run_interactive_mode(agent)
            
    except Exception as e:
        print(f"Error: {e}")
        if "GROQ_API_KEY" not in os.environ:
            print("\nPlease set your GROQ_API_KEY environment variable before running this script.")
            print("You can do this by either:")
            print("1. Running: export GROQ_API_KEY='your-api-key' (on Linux/Mac)")
            print("2. Running: set GROQ_API_KEY=your-api-key (on Windows)")
            print("3. Creating a .env file with GROQ_API_KEY=your-api-key")

if __name__ == "__main__":
    main()
