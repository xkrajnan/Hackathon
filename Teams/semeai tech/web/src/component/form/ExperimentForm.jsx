import * as React from "react";
import PropTypes from 'prop-types';
import { Typography, TextField, Button, Checkbox, FormControlLabel, withStyles } from "@material-ui/core";
import {Link} from "react-router-dom";
import * as Text from 'web-constant/text';
import webStyle from 'web-style/webStyle';
import axios from 'axios'


class ExperimentForm extends React.Component {

  constructor(props) {
    super(props);

    this.state = {
      inputBrandStory: '',
      generatedSoFar: ''
    };
  }

  handleInput(event) {
    this.setState({inputBrandStory: event.target.value});
  }

  generateBrandStory(moreWords) {
    let inp = this.state.inputBrandStory;
    let reqBody = {inputBrandStory: inp, generateMore: moreWords};

    axios.post('http://localhost:5000/api/generateBS', reqBody)
      .then(response => {
        if (response.status == 200) {
           let textSoFar = this.state.generatedSoFar + ' ' + inp + ' ' + response.data.generatedBS;
           this.setState({generatedSoFar: textSoFar.trim()});
        }

        this.setState({inputBrandStory: ''});
      })
      .catch(error => {
        console.log(error.response);
        this.setState({inputBrandStory: ''});
      });
  }

  letUserEdit() {
    let textSoFar = this.state.generatedSoFar.trim();
    this.setState({inputBrandStory: textSoFar});
    this.setState({generatedSoFar: ''});
  }

  render() {
    return (
      <React.Fragment>
        <Typography variant="headline">{Text.DEMO_TITLE_TEXT}</Typography>
        <form>
          <section>
            <TextField
              id="outlined-multiline-flexible"
              label="Your BS"
              multiline
              value={this.state.inputBrandStory}
              style={{minWidth: '70em'}}
              inputProps={{style: {fontSize: 20}}}
              InputLabelProps={{style: {fontSize: 20}}}
              rowsMax="20"
              onChange={event => this.handleInput(event)}
              margin="normal"
              variant="outlined"
            />
          </section>

          <section>
            <Button
              variant="contained"
              color="primary"
              style={{width: '20em', marginLeft: '10em'}}
              disabled={this.state.inputBrandStory.length < 1}
              onClick={() => this.generateBrandStory(false)}>
                What now?
            </Button>

            <Button
              variant="contained"
              color="primary"
              style={{width: '20em', marginLeft: '20em'}}
              disabled={this.state.inputBrandStory.length < 1}
              onClick={() => this.generateBrandStory(true)}>
                I'm stuck! Generate more BS!
            </Button>
          </section>

          <section>
            <TextField
              id="outlined-multiline-flexible-output"
              label="Your Brand's Story So Far..."
              multiline
              value={this.state.generatedSoFar}
              disabled={true}
              rowsMax="20"
              margin="normal"
              inputProps={{style: {fontSize: 20}}}
              InputLabelProps={{style: {fontSize: 20}}}
              variant="outlined"
            />
          </section>

          <section>
            <Button
              variant="contained"
              color="primary"
              style={{width: '20em', marginLeft: '30em'}}
              disabled={this.state.generatedSoFar.length < 1}
              onClick={() => this.letUserEdit()}>
                More BS! MOOOOORE!
            </Button>
          </section>
        </form>
      </React.Fragment>
    );
  }
}

ExperimentForm.propTypes = {
  classes: PropTypes.shape({
    inpField: PropTypes.string
  }),
  path: PropTypes.string
};

export default withStyles(webStyle)(ExperimentForm);
