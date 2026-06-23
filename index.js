const jsPsych = initJsPsych({
  on_finish: function() {
    jsPsych.data.displayData();
  }
});

const welcome = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: '<p>Welcome to the V-MoE user study (test deployment)</p><p>Press any key to continue.</p>'
};

const dummy_question = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: '<p>This is a dummy stimulus.</p><p>Press SPACE to finish.</p>',
  choices: [' ']
};

const goodbye = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: '<p>Thanks! Pipeline works. Press any key to end.</p>'
};

jsPsych.run([welcome, dummy_question, goodbye]);
