import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import random

class TaskManager:
    """Class for managing tasks."""

    def __init__(self):
        """Initialize the TaskManager object."""
        self.tasks = pd.DataFrame(columns=['description', 'priority'])
        self.load_tasks()
        self.train_model()

    def load_tasks(self):
        """Load tasks from a CSV file if it exists."""
        try:
            self.tasks = pd.read_csv('tasks.csv')
        except FileNotFoundError:
            pass

    def save_tasks(self):
        """Save tasks to a CSV file."""
        self.tasks.to_csv('tasks.csv', index=False)

    def train_model(self):
        """Train the machine learning model for task prioritization."""
        vectorizer = CountVectorizer(stop_words='english', min_df=1)
        clf = MultinomialNB()
        self.model = make_pipeline(vectorizer, clf)
        self.model.fit(self.tasks['description'], self.tasks['priority'])

    def add_task(self, description, priority):
        """
        Add a new task to the task list.

        Parameters:
        - description (str): Description of the task.
        - priority (str): Priority of the task (Low/Medium/High).
        """
        new_task = pd.DataFrame({'description': [description], 'priority': [priority]})
        self.tasks = pd.concat([self.tasks, new_task], ignore_index=True)
        self.save_tasks()
        self.train_model()

    def remove_task(self, description):
        """
        Remove a task from the task list.

        Parameters:
        - description (str): Description of the task to be removed.
        """
        self.tasks = self.tasks[self.tasks['description'] != description]
        self.save_tasks()

    def list_tasks(self):
        """List all tasks currently in the task list."""
        if self.tasks.empty:
            print("No tasks available.")
        else:
            print(self.tasks)

    def prioritize_tasks(self):
        """Prioritize tasks based on their priority level."""
        self.tasks.sort_values(by='priority', inplace=True)
        self.save_tasks()

    def recommend_task(self):
        """Recommend a task based on machine learning predictions."""
        if self.tasks.empty:
            print("No tasks available for recommendations.")
        else:
            predictions = self.model.predict(self.tasks['description'])
            recommended_task = random.choice(self.tasks[self.tasks['priority'] == predictions[0]]['description'])
            print(f"Recommended task: {recommended_task} - Priority: {predictions[0]}")

    def main_menu(self):
        """Display the main menu and handle user input."""
        while True:
            print("\nTask Management App")
            print("1. Add Task")
            print("2. Remove Task")
            print("3. List Tasks")
            print("4. Prioritize Tasks")
            print("5. Recommend Task")
            print("6. Exit")

            choice = input("Select an option: ")

            if choice == "1":
                description = input("Enter task description: ")
                priority = input("Enter task priority (Low/Medium/High): ").capitalize()
                if priority not in ['Low', 'Medium', 'High']:
                    print("Invalid priority! Please enter Low, Medium, or High.")
                else:
                    self.add_task(description, priority)
                    print("Task added successfully.")

            elif choice == "2":
                description = input("Enter task description to remove: ")
                self.remove_task(description)
                print("Task removed successfully.")

            elif choice == "3":
                self.list_tasks()

            elif choice == "4":
                self.prioritize_tasks()
                print("Tasks prioritized successfully.")

            elif choice == "5":
                self.recommend_task()

            elif choice == "6":
                print("Goodbye!")
                break

            else:
                print("Invalid option. Please select a valid option.")

if __name__ == "__main__":
    manager = TaskManager()
    manager.main_menu()
