import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ClaimUploader } from "@/components/ClaimUploader";
import { ClaimsTable } from "@/components/ClaimsTable";
import { AnalyticsDashboard } from "@/components/AnalyticsDashboard";
import { Shield, AlertTriangle, CheckCircle, TrendingUp } from "lucide-react";

const Index = () => {
  const [activeTab, setActiveTab] = useState("dashboard");

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-gradient-to-br from-primary to-success rounded-lg shadow-lg">
                <Shield className="h-6 w-6 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-foreground">FraudGuard</h1>
                <p className="text-sm text-muted-foreground">Insurance Fraud Detection System</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 px-3 py-1 bg-success/10 rounded-full">
                <CheckCircle className="h-4 w-4 text-success" />
                <span className="text-sm font-medium text-success">System Active</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-4 bg-card/50 backdrop-blur-sm">
            <TabsTrigger value="dashboard" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              Dashboard
            </TabsTrigger>
            <TabsTrigger value="upload" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              Upload Claims
            </TabsTrigger>
            <TabsTrigger value="claims" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              Claims Review
            </TabsTrigger>
            <TabsTrigger value="analytics" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              Analytics
            </TabsTrigger>
          </TabsList>

          {/* Dashboard Overview */}
          <TabsContent value="dashboard" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <Card className="bg-gradient-to-br from-card to-card/50 border-border/50 shadow-lg">
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Claims</CardTitle>
                  <TrendingUp className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">2,847</div>
                  <p className="text-xs text-muted-foreground">
                    +12.5% from last month
                  </p>
                </CardContent>
              </Card>
              
              <Card className="bg-gradient-to-br from-danger/10 to-danger/5 border-danger/20 shadow-lg">
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Fraudulent Claims</CardTitle>
                  <AlertTriangle className="h-4 w-4 text-danger" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-danger">127</div>
                  <p className="text-xs text-muted-foreground">
                    4.46% fraud rate
                  </p>
                </CardContent>
              </Card>

              <Card className="bg-gradient-to-br from-success/10 to-success/5 border-success/20 shadow-lg">
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Verified Claims</CardTitle>
                  <CheckCircle className="h-4 w-4 text-success" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-success">2,720</div>
                  <p className="text-xs text-muted-foreground">
                    95.54% success rate
                  </p>
                </CardContent>
              </Card>

              <Card className="bg-gradient-to-br from-warning/10 to-warning/5 border-warning/20 shadow-lg">
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Amount Saved</CardTitle>
                  <Shield className="h-4 w-4 text-warning" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-warning">₹2.4M</div>
                  <p className="text-xs text-muted-foreground">
                    Fraud amount prevented
                  </p>
                </CardContent>
              </Card>
            </div>

            <AnalyticsDashboard />
          </TabsContent>

          {/* Upload Claims */}
          <TabsContent value="upload" className="space-y-6">
            <div className="max-w-2xl mx-auto">
              <Card className="bg-gradient-to-br from-card to-card/50 border-border/50 shadow-xl">
                <CardHeader>
                  <CardTitle className="text-xl">Upload Insurance Claims</CardTitle>
                  <CardDescription>
                    Upload PDF documents or images of hospital bills for automated fraud detection
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ClaimUploader />
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Claims Review */}
          <TabsContent value="claims" className="space-y-6">
            <Card className="bg-gradient-to-br from-card to-card/50 border-border/50 shadow-lg">
              <CardHeader>
                <CardTitle className="text-xl">Claims Review</CardTitle>
                <CardDescription>
                  Monitor and review all insurance claims with fraud detection results
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ClaimsTable />
              </CardContent>
            </Card>
          </TabsContent>

          {/* Analytics */}
          <TabsContent value="analytics" className="space-y-6">
            <AnalyticsDashboard detailed />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
};

export default Index;