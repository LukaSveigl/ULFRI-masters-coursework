<script lang="ts">
    import AppIcon from "$lib/components/AppIcon.svelte";
    import DashboardLoanIndicator from "$lib/components/DashboardLoanIndicator.svelte";
    import MoneyIn from "$lib/components/icons/MoneyIn.svelte";
    import MoneyOut from "$lib/components/icons/MoneyOut.svelte";
    import { Button } from "$lib/shadcn/components/ui/button";
    import { HandCoins, ScrollText, Settings, User, Users } from "lucide-svelte";

    let { data } = $props();
</script>

<div class="grid grid-cols-[auto_auto] gap-6">
    <DashboardLoanIndicator
        icon={MoneyOut}
        n={data.unpaidN}
        typeString="unpaid"
        value={-data.unpaidValue}
        class="bg-red-200 text-red-950"
    />
    <DashboardLoanIndicator
        icon={MoneyIn}
        n={data.owedN}
        typeString="owed"
        value={data.owedValue}
        class="bg-green-200 text-green-950"
    />
</div>

<Button class="shadow-md" href="/issueloan">
    <AppIcon icon={HandCoins} class="!size-6" />
    Issue Loan
</Button>

<Button class="shadow-md" href="/history">
    <AppIcon icon={ScrollText} class="!size-6" />
    View Loan Details And History
</Button>

<Button class="shadow-md" href="/friends">
    <AppIcon icon={User} class="!size-6" />
    Manage Friends
</Button>

<Button class="shadow-md" href="/groups">
    <AppIcon icon={Users} class="!size-6" />
    Manage Groups
</Button>

<Button class="shadow-md" href="/settings">
    <AppIcon icon={Settings} class="!size-6" />
    Settings
</Button>
