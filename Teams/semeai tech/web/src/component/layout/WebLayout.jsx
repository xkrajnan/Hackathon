import * as React from 'react';
import PropTypes from 'prop-types';
import ExperimentForm from 'web-component/form/ExperimentForm';
import {Fade, Paper, Typography, withStyles} from '@material-ui/core';
import webStyle from 'web-style/webStyle';
import * as Text from 'web-constant/text';

class WebLayout extends React.Component {
  constructor(props) {
    super(props);
  }


  render() {
    return(
      <React.Fragment>
        <div className={this.props.classes.mainSign}>
          <main className={this.props.classes.main}>
            <section className={this.props.classes.alignCenter} color="secondary">
                <Typography className={this.props.classes.bsGen} variant="headline">BS Generator</Typography>
            </section>
            <Fade in timeout={1000}>
              <Paper elevation={0} >
                <ExperimentForm />
              </Paper>
            </Fade>
          </main>
        </div>
      </React.Fragment>
    );
  }
}

WebLayout.propTypes = {
  classes: PropTypes.shape({
    bsGen: PropTypes.string,
    main: PropTypes.string,
    mainSign: PropTypes.string,
    alignCenter: PropTypes.string
  }),
  path: PropTypes.string
};

export default withStyles(webStyle)(WebLayout);
