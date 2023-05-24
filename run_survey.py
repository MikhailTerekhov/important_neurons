import tkinter as tk
import tkinter.messagebox as tkmb
from datetime import datetime

model_name = "InceptionV1"
layer_name = "mixed5a"
neurons = [0, 1, 2, 3]
# neurons = [0]

question_vars = []
cur_row = 1
cur_neuron_ind = 0
is_locked = False

first_radiobuttons = []
extra_radiobuttons = []

date_str = f'{datetime.now():%Y%m%d_%H%M%S}'
data_fname = f"data/answers_{date_str}.csv"
with open(data_fname, 'a') as file:
    file.write("model,layer,neuron,question1,text_question,question3,question4,question5,question6")


def lock_text_input():
    global is_locked, model_name, layer_name, neurons, cur_neuron_ind, cur_image, radiobuttons

    if not question_vars[0].get():
        tkmb.showwarning("Warning", "Please select a response to the first question before saving the cdescription.")
        return

    is_locked = True
    text_entry.config(state=tk.DISABLED)
    lock_button.config(state=tk.DISABLED)
    cur_image = tk.PhotoImage(file=f"data/visualizations/{layer_name}_{neurons[cur_neuron_ind]}.png")
    image_label.config(image=cur_image)

    for b in first_radiobuttons:
        b.config(state=tk.DISABLED)
    for b in extra_radiobuttons:
        b.config(state=tk.NORMAL)


def save_response():
    global is_locked, model_name, layer_name, neurons, cur_neuron_ind, window, cur_image

    if not is_locked:
        tkmb.showwarning("Warning", "Please save the text input before recording the response.")
        return

    if not question_vars[1].get():
        tkmb.showwarning("Warning", "Please select a response to Question #3 before submitting the answer.")
        return

    if not question_vars[2].get():
        tkmb.showwarning("Warning", "Please select a response to Question #4 before submitting the answer.")
        return

    if not question_vars[3].get():
        tkmb.showwarning("Warning", "Please select a response to Question #5 before submitting the answer.")
        return

    if not question_vars[4].get():
        tkmb.showwarning("Warning", "Please select a response to Question #6 before submitting the answer.")
        return


    # Get the selected response from the radiobuttons
    question_responses = [v.get() for v in question_vars]
    text_response = text_entry.get("1.0", tk.END).strip()

    # Save the responses locally
    with open(data_fname, 'a') as out_file:
        print(f"\n{model_name},{layer_name},{neurons[cur_neuron_ind]},"
                       f"{question_responses[0]},\"{text_response}\","
                       f"{question_responses[1]},{question_responses[2]},{question_responses[3]},{question_responses[4]}", file=out_file, flush=True)

    # Clear the radiobutton selection and text field
    for v in question_vars:
        v.set(0)
    text_entry.config(state=tk.NORMAL)
    lock_button.config(state=tk.NORMAL)
    text_entry.delete("1.0", tk.END)

    is_locked = False
    cur_neuron_ind += 1
    if cur_neuron_ind == len(neurons):
        tkmb.showinfo("Info", "You've completed the survey. Thank you for your participation!")
        window.destroy()
    else:
        # Update the image
        cur_image = tk.PhotoImage(file=f"data/visualizations/{layer_name}_{neurons[cur_neuron_ind]}_hidden.png")
        image_label.config(image=cur_image)
        image_desc.config(text=f"Current feature: {cur_neuron_ind+1}/{len(neurons)}")

        for b in first_radiobuttons:
            b.config(state=tk.NORMAL)
        for b in extra_radiobuttons:
            b.config(state=tk.DISABLED)


def add_question(text, is_first=True):
    global cur_row, question_vars, first_radiobuttons, extra_radiobuttons

    # Create the first question with radiobuttons
    question_label = tk.Label(window, text=text, wraplength=600, anchor="w", justify="left")
    question_label.grid(row=cur_row, column=0, sticky="W", padx=2)

    question_vars.append(tk.StringVar())
    for i in range(5):
        question_option = tk.Radiobutton(window, text=f"{i+1}", variable=question_vars[-1], value=i+1)
        question_option.grid(row=cur_row, column=i+1, pady=10)
        if is_first:
            first_radiobuttons.append(question_option)
        else:
            extra_radiobuttons.append(question_option)

    cur_row += 1


# Create the main window
window = tk.Tk()
window.title("Feature visualizations")
window.geometry("1500x800")

cur_image = tk.PhotoImage(file=f"data/visualizations/{layer_name}_{neurons[cur_neuron_ind]}_hidden.png")


# Create a separate frame for the image
image_frame = tk.Frame(window)
image_frame.grid(row=0, columnspan=20)

# Create the image label
name = f"data/visualizations/{layer_name}_{neurons[0]}_hidden.png"
print(f"{name=}")
image_label = tk.Label(image_frame, image=cur_image)
image_label.pack()
image_desc = tk.Label(window,
                      text=f"Current feature: {cur_neuron_ind+1}/{len(neurons)}",
                      wraplength=600, anchor="w", justify="left")
image_desc.grid(row=1, column=0, sticky="W")
cur_row += 1


add_question("On a scale from 1 to 5, how clear are the objects on the "
             "4 feature visualization images? If you can’t recognise any"
             " objects on the images, respond as 1:", is_first=True)

# Create the text question
text_question_label = tk.Label(window,
                               text="Which words or phrases are best describing objects on "
                                    "the feature visualization images. List 0-4 phrases through comma:",
                               wraplength=600, anchor="w", justify="left")
text_question_label.grid(row=cur_row, column=0, sticky="W")
text_entry = tk.Text(window, height=2)
text_entry.grid(row=cur_row, column=1, columnspan=2, padx=5, pady=5, sticky="w")

lock_button = tk.Button(window, text="Save", command=lock_text_input)
lock_button.grid(row=cur_row, column=4, columnspan=3, padx=5, pady=5, sticky="w")

cur_row += 1


add_question("On a scale from 1 to 5, how closely do the dataset examples resemble "
             "the phrases you put in the previous question? Respond 1 if you didn’t write any phrases:",
             is_first=False)

add_question("On a scale from 1 to 5, how coherent are the examples "
             "(i.e. 1=they have nothing in common, 5=they all show the same concept)?",
             is_first=False)

add_question("On a scale from 1 to 5, do you agree that feature visualizations and "
             "dataset examples show 2 distinct concepts?\n"
             "1 = \"There clearly aren’t two objects\"; 5=\"There clearly are two objects\"",
             is_first=False)

add_question("On a scale from 1 to 5, do you agree that feature visualizations and "
             "dataset examples show 3 distinct concepts?\n"
             "1 = \"There clearly aren’t three objects\"; 5=\"There clearly are three objects\"",
             is_first=False)

# Create the submit button
submit_button = tk.Button(window, text="Submit", command=save_response)
submit_button.grid(row=cur_row, columnspan=5, pady=10)

for b in first_radiobuttons:
    b.config(state=tk.NORMAL)
for b in extra_radiobuttons:
    b.config(state=tk.DISABLED)

# Start the main event loop
window.mainloop()
