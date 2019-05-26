import * as React from 'react';
import {BrowserRouter, Switch, Route} from 'react-router-dom';
import ApplicationRoute from 'web-component/route/ApplicationRoute';
import IndexRoute from 'web-component/route/IndexRoute';
import * as Routes from 'web-constant/route';

function ApplicationRouter(){
  return (
    <BrowserRouter>
      <Switch>
        <Route exact path={Routes.INDEX_ROUTE} component={IndexRoute} />
        <Route exact path={Routes.DEMO_ROUTE} component={ApplicationRoute} />
      </Switch>
    </BrowserRouter>
  );

}

export default ApplicationRouter;
