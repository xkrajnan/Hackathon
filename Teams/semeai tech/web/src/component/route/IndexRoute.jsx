import * as React from 'react';
import { Redirect } from 'react-router';
import {DEMO_ROUTE} from 'web-constant/route';

function IndexRoute(){
  return <Redirect to={DEMO_ROUTE} />;
}

export default IndexRoute;
