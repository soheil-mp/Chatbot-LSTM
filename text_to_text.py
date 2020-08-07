import numpy as np

# Iterate through the number of times you want to ask question
def text_to_text(input_text, enc_model , dec_model, str_to_tokens, preprocess_text, tokenizer, maxlen_answers):

    # Get the input and predict it with the encoder model
    states_values = enc_model.predict(str_to_tokens(preprocess_text(input_text)))

    # Initialize the target sequence with zero - array([[0.]])
    empty_target_seq = np.zeros(shape = (1, 1))

    # Update the target sequence with index of "start"
    empty_target_seq[0, 0] = tokenizer.word_index["start"]

    # Initialize the stop condition with False
    stop_condition = False

    # Initialize the decoded words with an empty string
    decoded_translation = ''

    # While stop_condition is false
    while not stop_condition :

        # Predict the (target sequence + the output from encoder model) with decoder model
        dec_outputs , h , c = dec_model.predict([empty_target_seq] + states_values)

        # Get the index for sampled word
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])

        # Initialize the sampled word with None
        sampled_word = None

        # Iterate through words and their indexes
        for word, index in tokenizer.word_index.items() :

            # If the index is equal to sampled word's index
            if sampled_word_index == index :

                # Add the word to the decoded string
                decoded_translation += ' {}'.format(word)

                # Update the sampled word
                sampled_word = word
        
        # If sampled word is equal to "end" OR the length of decoded string is more that what is allowed
        if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:

            # Make the stop_condition to true
            stop_condition = True
            
        # Initialize back the target sequence to zero - array([[0.]])    
        empty_target_seq = np.zeros(shape = (1, 1))  

        # Update the target sequence with index of "start"
        empty_target_seq[0, 0] = sampled_word_index

        # Get the state values
        states_values = [h, c] 

    # return the decoded string
    return decoded_translation[:-3]