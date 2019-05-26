import * as React from "react";
import { MuiThemeProvider, CssBaseline } from "@material-ui/core";
import applicationTheme from 'web-theme/applicationTheme';
import ApplicationRouter from 'web-router/ApplicationRouter';

function Application(){
  return (
    <MuiThemeProvider theme={applicationTheme}>
    <CssBaseline />
    <ApplicationRouter />
  </MuiThemeProvider>
  );
}

export default Application;
